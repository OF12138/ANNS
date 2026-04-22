// =============================================================================
// exp/pq_phase_timing.cc — Per-phase time breakdown for PQ_RERANK search
//
// Splits pq_flat_search_rerank into three timed phases for each query:
//   Phase 1  LUT build   — build the M×K distance table from the query
//   Phase 2  Coarse scan — scan all N base codes, keep top-p candidates
//   Phase 3  Rerank      — exact float32 IP for the p candidates, keep top-k
//
// Compile from the project root:
//   g++ exp/pq_phase_timing.cc -o exp/pq_phase_timing -O2 -std=c++11 -lpthread
//
// Run:
//   ./exp/pq_phase_timing
// =============================================================================

#include <vector>
#include <queue>
#include <utility>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sys/time.h>

#include "ARM/Alg_normal/pq_flat_normal.h"

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

static inline int64_t tv_diff_us(const struct timeval& a, const struct timeval& b)
{
    return (b.tv_sec * 1000000LL + b.tv_usec) - (a.tv_sec * 1000000LL + a.tv_usec);
}

template<typename T>
T* LoadData(const std::string& path, size_t& n, size_t& d)
{
    std::ifstream fin(path, std::ios::binary);
    fin.read((char*)&n, 4);
    fin.read((char*)&d, 4);
    T* data = new T[n * d];
    int sz = sizeof(T);
    for (size_t i = 0; i < n; ++i)
        fin.read((char*)data + i * d * sz, d * sz);
    std::cerr << "loaded " << path << "  n=" << n << "  d=" << d << "\n";
    return data;
}

// ----------------------------------------------------------------------------
// Instrumented PQ search: returns {lut_us, scan_us, rerank_us, recall}
// ----------------------------------------------------------------------------

struct PhaseResult
{
    int64_t lut_us;
    int64_t scan_us;
    int64_t rerank_us;
    float   recall;
};

PhaseResult pq_search_timed(const PQIndex& index, const float* base,
                             const float* query, const int* gt, size_t gt_d,
                             size_t k, size_t p)
{
    const size_t M    = index.M;
    const size_t K    = index.K;
    const size_t dsub = index.dsub;
    const size_t d    = index.vecdim;

    struct timeval t0, t1, t2, t3;

    // ── Phase 1: LUT build ────────────────────────────────────────────────────
    // For each (subspace m, centroid c): compute IP(query_m, centroid[m][c])
    gettimeofday(&t0, NULL);

    std::vector<float> dtable(M * K);
    for (size_t m = 0; m < M; ++m)
    {
        const float* q_m    = query + m * dsub;
        const float* c_base = index.centroids.data() + m * K * dsub;
        for (size_t c = 0; c < K; ++c)
        {
            const float* cv = c_base + c * dsub;
            float ip = 0.0f;
            for (size_t j = 0; j < dsub; ++j) ip += q_m[j] * cv[j];
            dtable[m * K + c] = ip;
        }
    }

    gettimeofday(&t1, NULL);

    // ── Phase 2: Coarse scan ──────────────────────────────────────────────────
    // Iterate all N base codes; accumulate approx IP via table lookup; keep top-p
    std::priority_queue<std::pair<float, uint32_t>> coarse_heap;

    for (size_t i = 0; i < index.base_number; ++i)
    {
        const uint8_t* code = index.codes.data() + i * M;
        float approx_ip = 0.0f;
        for (size_t m = 0; m < M; ++m) approx_ip += dtable[m * K + code[m]];
        float dis = 1.0f - approx_ip;

        if (coarse_heap.size() < p) {
            coarse_heap.push({dis, (uint32_t)i});
        } else if (dis < coarse_heap.top().first) {
            coarse_heap.push({dis, (uint32_t)i});
            coarse_heap.pop();
        }
    }

    std::vector<uint32_t> cand_ids;
    cand_ids.reserve(p);
    while (!coarse_heap.empty()) {
        cand_ids.push_back(coarse_heap.top().second);
        coarse_heap.pop();
    }

    gettimeofday(&t2, NULL);

    // ── Phase 3: Rerank ───────────────────────────────────────────────────────
    // Exact float32 IP for each candidate; keep top-k
    std::priority_queue<std::pair<float, uint32_t>> result;

    for (uint32_t id : cand_ids)
    {
        const float* bv = base + (size_t)id * d;
        float ip = 0.0f;
        for (size_t j = 0; j < d; ++j) ip += bv[j] * query[j];
        float dis = 1.0f - ip;

        if (result.size() < k) {
            result.push({dis, id});
        } else if (dis < result.top().first) {
            result.push({dis, id});
            result.pop();
        }
    }

    gettimeofday(&t3, NULL);

    // ── Recall ────────────────────────────────────────────────────────────────
    size_t acc = 0;
    while (!result.empty()) {
        uint32_t id = result.top().second; result.pop();
        for (size_t j = 0; j < k; ++j)
            if ((uint32_t)gt[j] == id) { ++acc; break; }
    }

    return {
        tv_diff_us(t0, t1),
        tv_diff_us(t1, t2),
        tv_diff_us(t2, t3),
        (float)acc / k
    };
}

// ----------------------------------------------------------------------------
// main
// ----------------------------------------------------------------------------

int main()
{
    const std::string data_path = "/anndata/";
    size_t test_n = 0, base_n = 0, gt_d = 0, vecdim = 0;

    float* queries = LoadData<float>(data_path + "DEEP100K.query.fbin",              test_n,  vecdim);
    int*   gt      = LoadData<int>  (data_path + "DEEP100K.gt.query.100k.top100.bin",test_n,  gt_d);
    float* base    = LoadData<float>(data_path + "DEEP100K.base.100k.fbin",          base_n,  vecdim);
    test_n = 2000;

    const size_t k = 10;

    // Build PQ index (scalar)
    struct timeval tb0, tb1;
    gettimeofday(&tb0, NULL);
    PQIndex pq_index;
    pq_index.build(base, base_n, vecdim);
    gettimeofday(&tb1, NULL);
    std::cerr << "[build] PQIndex: " << tv_diff_us(tb0, tb1) / 1000 << " ms\n\n";

    // p values to test
    const size_t p_vals[] = {100, 200, 400, 800};
    const int    n_p      = 4;

    // Header
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n";
    std::cout << "pq_flat_search_rerank — per-phase timing breakdown\n";
    std::cout << "N=100K  M=8  K=256  dsub=12  queries=2000  k=10\n\n";

    std::cout
        << std::setw(6)  << "p"
        << std::setw(10) << "recall"
        << std::setw(14) << "total(us)"
        << std::setw(14) << "lut(us)"
        << std::setw(14) << "scan(us)"
        << std::setw(14) << "rerank(us)"
        << std::setw(10) << "lut%"
        << std::setw(10) << "scan%"
        << std::setw(10) << "rerank%"
        << "\n";
    std::cout << std::string(106, '-') << "\n";

    for (int pi = 0; pi < n_p; ++pi)
    {
        const size_t p = p_vals[pi];

        double sum_lut = 0, sum_scan = 0, sum_rerank = 0, sum_recall = 0;

        for (size_t i = 0; i < test_n; ++i)
        {
            PhaseResult r = pq_search_timed(
                pq_index, base,
                queries + i * vecdim,
                gt      + i * gt_d,
                k, k, p);

            sum_lut    += r.lut_us;
            sum_scan   += r.scan_us;
            sum_rerank += r.rerank_us;
            sum_recall += r.recall;
        }

        double avg_lut    = sum_lut    / test_n;
        double avg_scan   = sum_scan   / test_n;
        double avg_rerank = sum_rerank / test_n;
        double avg_total  = avg_lut + avg_scan + avg_rerank;
        double avg_recall = sum_recall / test_n;

        std::cout
            << std::setw(6)  << p
            << std::setw(10) << avg_recall
            << std::setw(14) << avg_total
            << std::setw(14) << avg_lut
            << std::setw(14) << avg_scan
            << std::setw(14) << avg_rerank
            << std::setw(9)  << (avg_lut    / avg_total * 100) << "%"
            << std::setw(9)  << (avg_scan   / avg_total * 100) << "%"
            << std::setw(9)  << (avg_rerank / avg_total * 100) << "%"
            << "\n";
    }

    delete[] queries;
    delete[] gt;
    delete[] base;
    return 0;
}
