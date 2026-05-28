// =============================================================================
// ivf_flat_simd_mpi_omp.h — Hybrid MPI × OMP IVF-SIMD search (Plan II)
//
// Parallelization strategy: query-batch partition across MPI processes,
//   OMP-parallel fine scan within each process.
//
//   MPI layer  — rank r processes queries[ r*Q/P .. (r+1)*Q/P )
//     No per-query MPI communication. Each process works independently.
//     One MPI_Gather at the END of the batch to collect all results.
//
//   OMP layer  — within each query, T threads share the fine scan
//     Clusters are distributed with schedule(dynamic,1) for load balance.
//     Each thread maintains its own local heap; heaps are merged after the
//     parallel region.
//
// Comparison with Plan I (nprobe-MPI):
//   Plan I:  3 MPI calls per query (Bcast + 2×Gather).  No OMP.
//   Plan II: 2 MPI calls per BATCH (Gather recalls + Gather latencies).
//            OMP inside every query.  Communication cost is amortized over Q.
//
// Compile:  mpicxx main_mpi_omp.cc -o main_mpi_omp -O2 -std=c++11 -fopenmp -lm
// Run:      OMP_NUM_THREADS=T mpiexec -n P ./main_mpi_omp
//           where P × T <= total allocated cores
// =============================================================================
#pragma once
#include <omp.h>
#include <vector>
#include <queue>
#include <utility>
#include <algorithm>
#include <cfloat>
#include <cstdint>
#include "ivf_flat_simd.h"


// =============================================================================
// HybridTimings — per-query phase timing accumulators (seconds)
//   Accumulated over multiple queries; divide by query count for per-query avg.
//   Both fields are meaningful on every process.
// =============================================================================
struct HybridTimings {
    double t_coarse_s = 0.0;   // coarse phase: nlist centroid IPs + partial_sort
    double t_fine_s   = 0.0;   // fine phase: OMP parallel cluster scan + heap merge
};


// =============================================================================
// ivf_omp_search_query — single-query IVF search with OMP fine-scan
//
// INPUT:
//   idx      — IVFIndex (centroids, invlists), local to this process
//   base     — full base vectors [N × D]
//   query    — query vector [D]
//   k        — top-k results
//   nprobe   — clusters to probe
//   nthreads — OMP threads for fine scan (pass omp_get_max_threads())
//
// OUTPUT:
//   max-heap of (distance, vector_id), size k  — local top-k for this query
// =============================================================================
std::priority_queue<std::pair<float, uint32_t>>
ivf_omp_search_query(
    const IVFIndex& idx,
    const float*    base,
    const float*    query,
    size_t k, size_t nprobe,
    int nthreads)
{
    const size_t d     = idx.vecdim;
    const size_t nlist = idx.nlist;
    const size_t np    = std::min(nprobe, nlist);

    // ── Coarse phase (single thread) ─────────────────────────────────────────
    std::vector<uint32_t> probe_ids(np);
    {
        std::vector<std::pair<float, uint32_t>> coarse(nlist);
        for (size_t c = 0; c < nlist; ++c) {
            float ip = simd_inner_product_neon_unroll(
                idx.centroids.data() + c * d, query, d);
            coarse[c] = {1.0f - ip, static_cast<uint32_t>(c)};
        }
        std::partial_sort(coarse.begin(), coarse.begin() + np, coarse.end());
        for (size_t i = 0; i < np; ++i)
            probe_ids[i] = coarse[i].second;
    }

    // ── Fine scan (OMP parallel over clusters) ────────────────────────────────
    // Each thread owns a private heap. schedule(dynamic,1) balances unequal
    // cluster sizes: threads pick the next unclaimed cluster as they finish.
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> thread_heaps(nthreads);

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        auto& heap = thread_heaps[tid];

        #pragma omp for schedule(dynamic, 1)
        for (int probe = 0; probe < static_cast<int>(np); ++probe) {
            const uint32_t c = probe_ids[probe];
            for (uint32_t orig : idx.invlists[c]) {
                float ip  = simd_inner_product_neon_unroll(
                    base + static_cast<size_t>(orig) * d, query, d);
                float dis = 1.0f - ip;
                if (heap.size() < k)
                    heap.push({dis, orig});
                else if (dis < heap.top().first) {
                    heap.pop();
                    heap.push({dis, orig});
                }
            }
        }
    }

    // ── Merge thread heaps → global top-k ────────────────────────────────────
    std::priority_queue<std::pair<float, uint32_t>> result;
    for (auto& h : thread_heaps) {
        while (!h.empty()) {
            auto top = h.top(); h.pop();
            if (result.size() < k)
                result.push(top);
            else if (top.first < result.top().first) {
                result.pop();
                result.push(top);
            }
        }
    }
    return result;
}


// =============================================================================
// ivf_omp_search_query_timed — same with phase timers via omp_get_wtime()
//
// INPUT / OUTPUT: same as ivf_omp_search_query, plus:
//   tm  — pointer to HybridTimings; fields are *added to* on each call.
//         Caller must zero-initialise before the first call.
// =============================================================================
std::priority_queue<std::pair<float, uint32_t>>
ivf_omp_search_query_timed(
    const IVFIndex& idx,
    const float*    base,
    const float*    query,
    size_t k, size_t nprobe,
    int nthreads,
    HybridTimings* tm)
{
    const size_t d     = idx.vecdim;
    const size_t nlist = idx.nlist;
    const size_t np    = std::min(nprobe, nlist);

    // ── Coarse ────────────────────────────────────────────────────────────────
    std::vector<uint32_t> probe_ids(np);
    double t0 = omp_get_wtime();
    {
        std::vector<std::pair<float, uint32_t>> coarse(nlist);
        for (size_t c = 0; c < nlist; ++c) {
            float ip = simd_inner_product_neon_unroll(
                idx.centroids.data() + c * d, query, d);
            coarse[c] = {1.0f - ip, static_cast<uint32_t>(c)};
        }
        std::partial_sort(coarse.begin(), coarse.begin() + np, coarse.end());
        for (size_t i = 0; i < np; ++i)
            probe_ids[i] = coarse[i].second;
    }
    tm->t_coarse_s += omp_get_wtime() - t0;

    // ── Fine scan ─────────────────────────────────────────────────────────────
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> thread_heaps(nthreads);
    t0 = omp_get_wtime();

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        auto& heap = thread_heaps[tid];

        #pragma omp for schedule(dynamic, 1)
        for (int probe = 0; probe < static_cast<int>(np); ++probe) {
            const uint32_t c = probe_ids[probe];
            for (uint32_t orig : idx.invlists[c]) {
                float ip  = simd_inner_product_neon_unroll(
                    base + static_cast<size_t>(orig) * d, query, d);
                float dis = 1.0f - ip;
                if (heap.size() < k)
                    heap.push({dis, orig});
                else if (dis < heap.top().first) {
                    heap.pop();
                    heap.push({dis, orig});
                }
            }
        }
    }
    tm->t_fine_s += omp_get_wtime() - t0;

    // ── Merge ─────────────────────────────────────────────────────────────────
    std::priority_queue<std::pair<float, uint32_t>> result;
    for (auto& h : thread_heaps) {
        while (!h.empty()) {
            auto top = h.top(); h.pop();
            if (result.size() < k)
                result.push(top);
            else if (top.first < result.top().first) {
                result.pop();
                result.push(top);
            }
        }
    }
    return result;
}
