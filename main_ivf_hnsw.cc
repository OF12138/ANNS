// =============================================================================
// main_ivf_hnsw.cc -- IVF+HNSW two-level index, single-thread baseline
//
// Coarse: IVF centroid scan (nlist=1024, nprobe=16)
// Fine:   per-cluster HNSW beam search (M=16, ef_construction=200, ef=50)
//
// Compile:
//   g++ main_ivf_hnsw.cc -o main_ivf_hnsw -O2 -std=c++11 -lpthread -lm
// =============================================================================

#define IVF_NLIST            1024
#define IVF_NPROBE           16
#define HNSW_M               16
#define HNSW_EF_CONSTRUCTION 200
#define HNSW_EF_SEARCH       50

#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <iomanip>
#include <sys/time.h>

#include "ARM/Alg_parallel/ivf_hnsw.h"


// -----------------------------------------------------------------------------
// Data loader: binary format [n:4B][d:4B][n*d values of type T]
// -----------------------------------------------------------------------------
template<typename T>
T* LoadData(const std::string& path, size_t& n, size_t& d)
{
    std::ifstream fin(path, std::ios::binary);
    fin.read(reinterpret_cast<char*>(&n), 4);
    fin.read(reinterpret_cast<char*>(&d), 4);
    T* data = new T[n * d];
    for (size_t i = 0; i < n; ++i)
        fin.read(reinterpret_cast<char*>(data) + i * d * sizeof(T),
                 d * sizeof(T));
    fin.close();
    return data;
}

static inline int64_t tv_diff_us(const struct timeval& a,
                                  const struct timeval& b)
{
    return (b.tv_sec * 1000000LL + b.tv_usec)
         - (a.tv_sec * 1000000LL + a.tv_usec);
}


int main()
{
    // ── Config banner ─────────────────────────────────────────────────────────
    std::cerr << "========================================\n";
    std::cerr << "[config] IVF+HNSW  (single-thread baseline)\n";
    std::cerr << "[config] ivf_nlist="          << IVF_NLIST
              << "  ivf_nprobe="               << IVF_NPROBE << "\n";
    std::cerr << "[config] hnsw_M="             << HNSW_M
              << "  hnsw_ef_construction="     << HNSW_EF_CONSTRUCTION
              << "  hnsw_ef_search="           << HNSW_EF_SEARCH << "\n";
    std::cerr << "========================================\n";

    // ── Load data ─────────────────────────────────────────────────────────────
    const std::string data_path = "/anndata/";
    size_t test_number = 0, base_number = 0, test_gt_d = 0, vecdim = 0;

    float* test_query = LoadData<float>(data_path + "DEEP100K.query.fbin",
                                        test_number, vecdim);
    int*   test_gt    = LoadData<int>  (data_path + "DEEP100K.gt.query.100k.top100.bin",
                                        test_number, test_gt_d);
    float* base       = LoadData<float>(data_path + "DEEP100K.base.100k.fbin",
                                        base_number, vecdim);
    test_number = 2000;
    const size_t k = 10;

    std::cerr << "[data] base=" << base_number << "  queries=" << test_number
              << "  dim=" << vecdim << "\n";

    // ── Build IVF+HNSW ────────────────────────────────────────────────────────
    IVFHNSWIndex idx;

    struct timeval tb0, tb1;
    gettimeofday(&tb0, NULL);

    std::cerr << "[build] running IVF k-means (nlist=" << IVF_NLIST << ")...\n";
    ivf_hnsw_build(idx, base, base_number, vecdim,
                   IVF_NLIST, 25, HNSW_M, HNSW_EF_CONSTRUCTION);

    gettimeofday(&tb1, NULL);
    std::cerr << "[build] done: " << tv_diff_us(tb0, tb1) / 1000 << " ms\n";

    // ── Warm-up ───────────────────────────────────────────────────────────────
    std::cerr << "[warmup] running " << test_number << " queries...\n";
    for (size_t i = 0; i < test_number; ++i)
        ivf_hnsw_search(idx, test_query + i * vecdim,
                        k, IVF_NPROBE, HNSW_EF_SEARCH);

    // ── Measured run ──────────────────────────────────────────────────────────
    double total_recall  = 0.0;
    double total_lat_us  = 0.0;
    int64_t t_coarse_sum = 0, t_fine_sum = 0;

    struct timeval batch_t0, batch_t1;
    gettimeofday(&batch_t0, NULL);

    for (size_t i = 0; i < test_number; ++i) {
        const float* q = test_query + i * vecdim;

        struct timeval t0, t1, t2;
        gettimeofday(&t0, NULL);

        // Phase 1: coarse (measure separately)
        const size_t d     = idx.ivf.vecdim;
        const size_t nlist = idx.ivf.nlist;
        const size_t np    = std::min((size_t)IVF_NPROBE, nlist);
        std::vector<std::pair<float, uint32_t>> coarse(nlist);
        for (size_t c = 0; c < nlist; ++c) {
            float ip = simd_inner_product_neon_unroll(
                idx.ivf.centroids.data() + c * d, q, d);
            coarse[c] = {1.0f - ip, static_cast<uint32_t>(c)};
        }
        std::partial_sort(coarse.begin(), coarse.begin() + np, coarse.end());

        gettimeofday(&t1, NULL);
        t_coarse_sum += tv_diff_us(t0, t1);

        // Phase 2: HNSW fine search
        std::priority_queue<std::pair<float, uint32_t>> global_heap;
        for (size_t pi = 0; pi < np; ++pi) {
            uint32_t c    = coarse[pi].second;
            auto*    hnsw = idx.clusters[c];
            if (!hnsw) continue;
            auto local_res = hnsw_search_simd(hnsw, q, k, HNSW_EF_SEARCH);
            while (!local_res.empty()) {
                auto [dist, orig] = local_res.top();
                local_res.pop();
                if (global_heap.size() < k) {
                    global_heap.push({dist, orig});
                } else if (dist < global_heap.top().first) {
                    global_heap.pop();
                    global_heap.push({dist, orig});
                }
            }
        }

        gettimeofday(&t2, NULL);
        t_fine_sum  += tv_diff_us(t1, t2);
        total_lat_us += tv_diff_us(t0, t2);

        // Recall evaluation
        std::set<uint32_t> gtset;
        for (size_t j = 0; j < k; ++j)
            gtset.insert(static_cast<uint32_t>(
                test_gt[j + i * test_gt_d]));
        size_t acc = 0;
        while (!global_heap.empty()) {
            if (gtset.count(global_heap.top().second)) ++acc;
            global_heap.pop();
        }
        total_recall += static_cast<double>(acc) / k;
    }

    gettimeofday(&batch_t1, NULL);
    double batch_s = tv_diff_us(batch_t0, batch_t1) / 1e6;

    // ── Report ────────────────────────────────────────────────────────────────
    double avg_recall    = total_recall  / test_number;
    double avg_lat_us    = total_lat_us  / test_number;
    double avg_coarse_us = (double)t_coarse_sum / test_number;
    double avg_fine_us   = (double)t_fine_sum   / test_number;
    double throughput    = test_number / batch_s;
    double amortized_us  = 1000000.0   / throughput;

    std::cerr << std::fixed << std::setprecision(2);
    std::cerr << "\n[phase breakdown — avg per query]\n";
    std::cerr << "  coarse (centroid scan) : " << avg_coarse_us << " us\n";
    std::cerr << "  fine   (HNSW search)   : " << avg_fine_us   << " us\n";

    std::cerr << "\n[throughput]\n";
    std::cerr << "  batch wall time   : " << batch_s * 1000.0 << " ms\n";
    std::cerr << "  throughput        : " << throughput << " queries/sec\n";
    std::cerr << "  amortized latency : " << amortized_us << " us/query\n";

    std::cout << std::fixed << std::setprecision(5);
    std::cout << "average recall: "         << avg_recall   << "\n";
    std::cout << "average latency (us): "   << avg_lat_us   << "\n";
    std::cout << "amortized latency (us): " << amortized_us << "\n";
    std::cout << "throughput (q/s): "       << throughput   << "\n";

    ivf_hnsw_free(idx);
    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    return 0;
}
