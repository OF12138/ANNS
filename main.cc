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

// =============================================================================
//  EXPERIMENT CONFIGURATION — change SEARCH_ALG and BUILD_PQ, then recompile.
//
//  SEARCH_ALG values:
//   ── Exact flat scan ──────────────────────────────────────────────────────
//    1  FLAT_SCALAR      flat_search()                scalar, int loop, exact
//    2  FLAT_NORMAL      flat_search_normal()         scalar, size_t loop, exact
//    3  FLAT_SIMD        simd_flat_search()           NEON float32x4 1-acc, exact
//    4  FLAT_SIMD_UNROLL simd_flat_search_unroll()    NEON float32x4 4-acc, exact
//   ── Scalar Quantization (2-phase, coarse p candidates → exact rerank) ───
//    5  SQ_NORMAL        sq_flat_search_normal()      scalar ADC coarse + scalar rerank
//    6  SQ_SIMD          sq_flat_search_simd()        NEON ADC coarse (uint8→float) + NEON rerank
//    7  SQ_SDC           sq_flat_search_sdc()         NEON SDC coarse (vmull_u8 integer) + NEON rerank
//   ── Product Quantization (2-phase, coarse p candidates → exact rerank) ──
//    8  PQ_NORMAL        pq_flat_search_normal()      scalar ADC scan, no rerank
//    9  PQ_RERANK        pq_flat_search_rerank()      scalar ADC scan + scalar float rerank
//   10  PQ_FLAT_SIMD     pq_flat_search_rerank_flat_simd()          flat-SIMD LUT (1 acc/centroid)
//   11  PQ_CC_SIMD       pq_flat_search_rerank_cross_centroid_simd() CC-SIMD LUT (0 reductions)
//   12  PQ_CC_UNROLL     pq_flat_search_rerank_cc_unroll()           CC-SIMD LUT + 4× unroll
//
//  BUILD_PQ values (only used when SEARCH_ALG is 8–12; ignored otherwise):
//    1  PQ_BUILD_SCALAR   pq_index.build()                scalar k-means
//    2  PQ_BUILD_SIMD     pq_build_index_simd_blocked()   SIMD double-tiled (i_blk × k_blk)
// =============================================================================
#define FLAT_SCALAR            1
#define FLAT_NORMAL            2
#define FLAT_SIMD              3
#define FLAT_SIMD_UNROLL       4
#define SQ_NORMAL              5
#define SQ_SIMD                6
#define SQ_SDC                 7
#define PQ_NORMAL              8
#define PQ_RERANK              9
#define PQ_FLAT_SIMD          10
#define PQ_CC_SIMD            11
#define PQ_CC_UNROLL          12

#define PQ_BUILD_SCALAR        1
#define PQ_BUILD_SIMD          2

// ── SET YOUR EXPERIMENT HERE ─────────────────────────────────────────────────
#define SEARCH_ALG   SQ_SIMD
#define BUILD_PQ     PQ_BUILD_SIMD
// ─────────────────────────────────────────────────────────────────────────────

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
#include "hnswlib/hnswlib/hnswlib.h"
#include "ARM/Alg_normal/flat_scan.h"
#include "ARM/Alg_normal/flat_scan_normal.h"
#include "ARM/Alg_normal/sq_flat_normal.h"
#include "ARM/Alg_normal/pq_flat_normal.h"
#include "ARM/Alg_parallel/flat_simd.h"
#include "ARM/Alg_parallel/sq_flat_simd.h"
#include "ARM/Alg_parallel/pq_flat_simd.h"

using namespace hnswlib;

// Coarse candidate count for SQ/PQ rerank (SEARCH_ALG 5–12).
// Larger p → higher recall, higher latency.
static const size_t p = 200;

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
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i)
        fin.read(((char*)data + i*d*sz), d*sz);
    fin.close();

    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";

    return data;
}

// SearchResult — stores per-query benchmark metrics
struct SearchResult
{
    float recall;
    int64_t latency; // microseconds
};

// build_index — constructs and persists an HNSW index (example, disabled by default)
void build_index(float* base, size_t base_number, size_t vecdim)
{
    const int efConstruction = 150;
    const int M = 16;

    HierarchicalNSW<float> *appr_alg;
    InnerProductSpace ipspace(vecdim);
    appr_alg = new HierarchicalNSW<float>(&ipspace, base_number, M, efConstruction);

    appr_alg->addPoint(base, 0);
    #pragma omp parallel for
    for(int i = 1; i < base_number; ++i)
        appr_alg->addPoint(base + 1ll*vecdim*i, i);

    char path_index[1024] = "files/hnsw.index";
    appr_alg->saveIndex(path_index);
}

// Returns elapsed microseconds between two gettimeofday snapshots.
static inline int64_t tv_diff_us(const struct timeval& a, const struct timeval& b)
{
    return (b.tv_sec * 1000000LL + b.tv_usec) - (a.tv_sec * 1000000LL + a.tv_usec);
}

int main(int argc, char *argv[])
{
    // ── Print experiment configuration ───────────────────────────────────────
    // These names mirror the comment table above; shown in PBS stdout for easy
    // identification when comparing multiple job outputs.
    static const char* search_names[] = 
    {
        "",
        "flat_search (scalar exact)",                                    //  1
        "flat_search_normal (scalar exact, size_t)",                     //  2
        "simd_flat_search (NEON 1-acc exact)",                           //  3
        "simd_flat_search_unroll (NEON 4-acc exact)",                    //  4
        "sq_flat_search_normal (SQ scalar ADC)",                         //  5
        "sq_flat_search_simd (SQ NEON ADC float)",                       //  6
        "sq_flat_search_sdc (SQ NEON SDC integer)",                      //  7
        "pq_flat_search_normal (PQ scalar ADC no-rerank)",               //  8
        "pq_flat_search_rerank (PQ scalar ADC + rerank)",                //  9
        "pq_flat_search_rerank_flat_simd (PQ flat-SIMD LUT)",            // 10
        "pq_flat_search_rerank_cross_centroid_simd (PQ CC-SIMD LUT)",    // 11
        "pq_flat_search_rerank_cc_unroll (PQ CC-SIMD 4x-unroll LUT)",   // 12
    };
    static const char* build_names[] = {
        "",
        "pq_index.build (scalar k-means)",                               //  1
        "pq_build_index_simd_blocked (SIMD double-tiled i×k)",           //  2
    };

    std::cerr << "========================================\n";
    std::cerr << "[config] search_alg = " << SEARCH_ALG
              << "  " << search_names[SEARCH_ALG] << "\n";
#if SEARCH_ALG >= PQ_NORMAL
    std::cerr << "[config] build_pq   = " << BUILD_PQ
              << "  " << build_names[BUILD_PQ] << "\n";
#endif
    std::cerr << "[config] p=" << p << "  k=10\n";
    std::cerr << "========================================\n";

    // ── Load dataset ─────────────────────────────────────────────────────────
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "/anndata/";
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin",              test_number, vecdim);
    auto test_gt    = LoadData<int>  (data_path + "DEEP100K.gt.query.100k.top100.bin",test_number, test_gt_d);
    auto base       = LoadData<float>(data_path + "DEEP100K.base.100k.fbin",          base_number, vecdim);
    test_number = 2000;

    const size_t k = 10;

    std::vector<SearchResult> results(test_number);

    // ── Index build (only constructs what SEARCH_ALG requires) ───────────────
    // Build time is printed to stderr so it appears in the PBS job's stderr
    // file separately from the recall/latency output in stdout.
    struct timeval tb0, tb1;

#if SEARCH_ALG >= SQ_NORMAL && SEARCH_ALG <= SQ_SDC
    SQIndex sq_index;
    gettimeofday(&tb0, NULL);
    sq_index.build(base, base_number, vecdim);
    gettimeofday(&tb1, NULL);
    std::cerr << "[build] SQIndex: " << tv_diff_us(tb0, tb1) / 1000 << " ms\n";
#endif

#if SEARCH_ALG >= PQ_NORMAL
    PQIndex pq_index;
    gettimeofday(&tb0, NULL);
#if   BUILD_PQ == PQ_BUILD_SCALAR
    pq_index.build(base, base_number, vecdim);
#elif BUILD_PQ == PQ_BUILD_SIMD
    pq_build_index_simd_blocked(pq_index, base, base_number, vecdim);
#else
    #error "Unknown BUILD_PQ value. Use PQ_BUILD_SCALAR or PQ_BUILD_SIMD."
#endif
    gettimeofday(&tb1, NULL);
    std::cerr << "[build] PQIndex (build_pq=" << BUILD_PQ << "): "
              << tv_diff_us(tb0, tb1) / 1000 << " ms\n";

    // Transposed centroid layout — only needed by CC-SIMD variants (11, 12)
#if SEARCH_ALG == PQ_CC_SIMD || SEARCH_ALG == PQ_CC_UNROLL
    PQIndexSIMD pq_simd(pq_index);
#endif
#endif

    // ── Query loop ────────────────────────────────────────────────────────────
    for(int i = 0; i < (int)test_number; ++i)
    {
        struct timeval val, newVal;
        gettimeofday(&val, NULL);

#if   SEARCH_ALG == FLAT_SCALAR
        auto res = flat_search(base, test_query + i*vecdim, base_number, vecdim, k);
#elif SEARCH_ALG == FLAT_NORMAL
        auto res = flat_search_normal(base, test_query + i*vecdim, base_number, vecdim, k);
#elif SEARCH_ALG == FLAT_SIMD
        auto res = simd_flat_search(base, test_query + i*vecdim, base_number, vecdim, k);
#elif SEARCH_ALG == FLAT_SIMD_UNROLL
        auto res = simd_flat_search_unroll(base, test_query + i*vecdim, base_number, vecdim, k);
#elif SEARCH_ALG == SQ_NORMAL
        auto res = sq_flat_search_normal(sq_index, base, test_query + i*vecdim, k, p);
#elif SEARCH_ALG == SQ_SIMD
        auto res = sq_flat_search_simd(sq_index, base, test_query + i*vecdim, k, p);
#elif SEARCH_ALG == SQ_SDC
        auto res = sq_flat_search_sdc(sq_index, base, test_query + i*vecdim, k, p);
#elif SEARCH_ALG == PQ_NORMAL
        auto res = pq_flat_search_normal(pq_index, test_query + i*vecdim, k);
#elif SEARCH_ALG == PQ_RERANK
        auto res = pq_flat_search_rerank(pq_index, base, test_query + i*vecdim, k, p);
#elif SEARCH_ALG == PQ_FLAT_SIMD
        auto res = pq_flat_search_rerank_flat_simd(pq_index, base, test_query + i*vecdim, k, p);
#elif SEARCH_ALG == PQ_CC_SIMD
        auto res = pq_flat_search_rerank_cross_centroid_simd(pq_simd, base, test_query + i*vecdim, k, p);
#elif SEARCH_ALG == PQ_CC_UNROLL
        auto res = pq_flat_search_rerank_cc_unroll(pq_simd, base, test_query + i*vecdim, k, p);
#else
        #error "Unknown SEARCH_ALG value. Set it to one of the defined constants (1–12)."
#endif

        gettimeofday(&newVal, NULL);
        int64_t diff = tv_diff_us(val, newVal);

        std::set<uint32_t> gtset;
        for(int j = 0; j < (int)k; ++j)
            gtset.insert(test_gt[j + i*test_gt_d]);

        size_t acc = 0;
        while (res.size())
        {
            if(gtset.count(res.top().second)) ++acc;
            res.pop();
        }
        results[i] = {(float)acc / k, diff};
    }

    // ── Report ────────────────────────────────────────────────────────────────
    float avg_recall = 0, avg_latency = 0;
    for(int i = 0; i < (int)test_number; ++i)
    {
        avg_recall  += results[i].recall;
        avg_latency += results[i].latency;
    }
    // 浮点误差可能导致一些精确算法平均recall不是1
    std::cout << "average recall: "       << avg_recall  / test_number << "\n";
    std::cout << "average latency (us): " << avg_latency / test_number << "\n";
    return 0;
}
