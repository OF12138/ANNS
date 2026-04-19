// =============================================================================
// main.cc — Benchmark harness for Approximate Nearest Neighbor (ANN) search
//
// Purpose:
//   This is the ONLY file you should modify for the assignment.
//   It loads the DEEP100K dataset, runs a search function for each of 2000
//   query vectors, computes recall against ground-truth labels, and reports
//   average recall and per-query latency in microseconds.
//
// Task:
//   Replace (or augment) the `flat_search` call with a faster approximate
//   algorithm (e.g., HNSW, SIMD-accelerated flat scan, IVF, etc.) while
//   maintaining high recall (ideally ≥ 0.9).
//
// Constraints:
//   - Do NOT modify flat_scan.h.
//   - The search function's return type must remain:
//       std::priority_queue<std::pair<float, uint32_t>>
//   - Index files must be saved under files/ (PBS cluster path limit).
//   - Do not build the index during the timed query loop (causes jitter).
//
// Data files (on the cluster at /anndata/):
//   DEEP100K.base.100k.fbin        — 100,000 base vectors (float32, dim=96)
//   DEEP100K.query.fbin            — query vectors (float32, dim=96)
//   DEEP100K.gt.query.100k.top100.bin — ground-truth top-100 labels (int32)
//
// Binary file format (.fbin / .bin):
//   [4 bytes: n (number of vectors)] [4 bytes: d (dimension)]
//   [n × d × sizeof(T) bytes: row-major vector data]
// =============================================================================

#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include <omp.h>
#include <immintrin.h>        // AVX2 / FMA intrinsics (_mm256_*, _mm_*)
#include "hnswlib/hnswlib/hnswlib.h"
#include "flat_scan.h"

using namespace hnswlib;

// =============================================================================
// SIMD-accelerated flat scan (Stage 1 — Flat-SIMD)
//
// Replaces the scalar flat_search with AVX2 vectorized dot products.
// Requires compiler flags: -mavx2 -mfma  (add -mfma alongside -mavx2 in qsub)
// =============================================================================

// simd_inner_product — dot product of two float32 vectors using AVX2 + FMA
//
// Processes 8 floats per SIMD lane per iteration.
// dim=96 is divisible by 8, so the loop covers all elements with zero remainder.
//
// Layout: a[0..dim-1], b[0..dim-1] — contiguous float arrays.
inline float simd_inner_product(const float* a, const float* b, size_t dim)
{
    // 256-bit accumulator holding 8 partial sums (one per float lane), init 0
    __m256 acc = _mm256_setzero_ps();

    for (size_t d = 0; d < dim; d += 8) 
    {
        // Load 8 floats from each vector; loadu = unaligned load (no alignment req)
        __m256 va = _mm256_loadu_ps(a + d);
        __m256 vb = _mm256_loadu_ps(b + d);

        // Fused multiply-add: acc = acc + va * vb  (single instruction, 1 cycle)
        // _mm256_fmadd_ps(x, y, z) computes x*y + z without intermediate rounding
        acc = _mm256_fmadd_ps(va, vb, acc);
    }

    // Horizontal reduce: sum 8 float lanes in the 256-bit register down to 1 scalar
    //
    // Step 1: split 256-bit into two 128-bit halves and add them pairwise
    //   acc  = [a0 a1 a2 a3 | a4 a5 a6 a7]
    //   lo   = [a0 a1 a2 a3]
    //   hi   = [a4 a5 a6 a7]
    //   s128 = [a0+a4, a1+a5, a2+a6, a3+a7]
    __m128 lo   = _mm256_castps256_ps128(acc);       // lower 128 bits (zero-cost cast)
    __m128 hi   = _mm256_extractf128_ps(acc, 1);     // upper 128 bits
    __m128 s128 = _mm_add_ps(lo, hi);                // 4-lane pairwise add

    // Step 2: two horizontal adds collapse 4 lanes → 2 → 1
    //   hadd(x, x) = [x0+x1, x0+x1, x2+x3, x2+x3]
    __m128 s2 = _mm_hadd_ps(s128, s128);
    __m128 s1 = _mm_hadd_ps(s2,   s2);

    return _mm_cvtss_f32(s1);  // extract lane 0 — the final scalar dot product
}

// simd_flat_search — exhaustive k-NN using SIMD dot products
//
// Drop-in replacement for flat_search; return type is identical.
// The max-heap logic is unchanged — only the distance kernel is vectorized.
std::priority_queue<std::pair<float, uint32_t>>
simd_flat_search(float* base, float* query,
                 size_t base_number, size_t vecdim, size_t k)
{
    std::priority_queue<std::pair<float, uint32_t>> q;

    for (size_t i = 0; i < base_number; ++i) 
    {
        // Compute IP distance = 1 - dot(base_i, query) via AVX2
        float dot = simd_inner_product(base + i * vecdim, query, vecdim);
        float dis = 1.0f - dot;

        if (q.size() < k) {
            q.push({dis, (uint32_t)i});
        } else if (dis < q.top().first) {
            // New vector is closer than the current farthest; swap it in
            q.push({dis, (uint32_t)i});
            q.pop();
        }
    }
    return q;
}

// LoadData<T> — reads a binary vector file into a flat array
//
// File format: 4-byte n, 4-byte d, then n*d values of type T.
// Returns a heap-allocated array of size n*d; caller owns the memory.
// Sets n and d as output parameters.
template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n,4);   // number of vectors
    fin.read((char*)&d,4);   // dimension per vector
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i)
    {
        // Read each row independently so different T sizes work correctly.
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();

    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";

    return data;
}

// SearchResult — stores per-query benchmark metrics
struct SearchResult
{
    float recall;
    int64_t latency; // 单位us
};

// build_index — constructs and persists an HNSW index (example, disabled by default)
//
// Parameters:
//   base        : base dataset, row-major [base_number × vecdim]
//   base_number : number of vectors
//   vecdim      : vector dimension
//
// HNSW hyperparameters:
//   efConstruction (150): search width during index build. Higher = better
//     recall at the cost of longer build time. Keep ≤ 200 for this dataset.
//   M (16): max number of bidirectional links per node per layer. Higher =
//     better recall / faster search but larger memory footprint. Keep ≤ 16.
//
// The first vector (index 0) must be added outside the parallel region because
// it sets the HNSW entry point; all subsequent vectors can be added in parallel.
void build_index(float* base, size_t base_number, size_t vecdim)
{
    const int efConstruction = 150; // 为防止索引构建时间过长，efc建议设置200以下
    const int M = 16; // M建议设置为16以下

    HierarchicalNSW<float> *appr_alg;
    InnerProductSpace ipspace(vecdim);
    // Construct empty HNSW graph with capacity base_number.
    appr_alg = new HierarchicalNSW<float>(&ipspace, base_number, M, efConstruction);

    // Insert vector 0 serially to establish the graph entry point.
    appr_alg->addPoint(base, 0);
    // Insert remaining vectors in parallel; hnswlib is thread-safe for addPoint.
    #pragma omp parallel for
    for(int i = 1; i < base_number; ++i) {
        appr_alg->addPoint(base + 1ll*vecdim*i, i);
    }

    char path_index[1024] = "files/hnsw.index";
    appr_alg->saveIndex(path_index);
}


int main(int argc, char *argv[])
{
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    // Load all three dataset files. LoadData infers n and d from the file header.
    std::string data_path = "/anndata/";
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    // Ground-truth: for query i, the top-k labels are at test_gt[i*test_gt_d .. i*test_gt_d+k-1].
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    // 只测试前2000条查询
    test_number = 2000;

    const size_t k = 10;  // Number of nearest neighbors to retrieve.

    std::vector<SearchResult> results;
    results.resize(test_number);

    // 如果你需要保存索引，可以在这里添加你需要的函数，你可以将下面的注释删除来查看pbs是否将build.index返回到你的files目录中
    // 要保存的目录必须是files/*
    // 每个人的目录空间有限，不需要的索引请及时删除，避免占空间太大
    // 不建议在正式测试查询时同时构建索引，否则性能波动会较大
    // 下面是一个构建hnsw索引的示例
    // build_index(base, base_number, vecdim);


    // -------------------------------------------------------------------------
    // Query loop — timed per query with gettimeofday (microsecond resolution)
    // -------------------------------------------------------------------------
    for(int i = 0; i < test_number; ++i)
    {
        const unsigned long Converter = 1000 * 1000;  // seconds → microseconds
        struct timeval val;
        int ret = gettimeofday(&val, NULL);  // start timer

        // 该文件已有代码中你只能修改该函数的调用方式
        // 可以任意修改函数名，函数参数或者改为调用成员函数，但是不能修改函数返回值。
        // REPLACE THIS CALL with your optimized search function.
        // The return type (max-heap of <distance, index> pairs) must not change.
        auto res = simd_flat_search(base, test_query + i*vecdim, base_number, vecdim, k);

        struct timeval newVal;
        ret = gettimeofday(&newVal, NULL);  // stop timer
        // Compute elapsed time in microseconds.
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

        // Build a set of ground-truth IDs for query i (top-k from the gt file).
        std::set<uint32_t> gtset;
        for(int j = 0; j < k; ++j)
        {
            int t = test_gt[j + i*test_gt_d];
            gtset.insert(t);
        }

        // Count how many of the returned k results appear in the ground truth.
        size_t acc = 0;
        while (res.size()) 
        {
            int x = res.top().second;
            if(gtset.find(x) != gtset.end()){
                ++acc;
            }
            res.pop();
        }
        // Recall@k = (# returned results that are true neighbors) / k
        float recall = (float)acc/k;

        results[i] = {recall, diff};
    }

    // Aggregate and print average recall and average latency.
    float avg_recall = 0, avg_latency = 0;
    for(int i = 0; i < test_number; ++i) 
    {
        avg_recall += results[i].recall;
        avg_latency += results[i].latency;
    }

    // 浮点误差可能导致一些精确算法平均recall不是1
    std::cout << "average recall: "<<avg_recall / test_number<<"\n";
    std::cout << "average latency (us): "<<avg_latency / test_number<<"\n";
    return 0;
}
