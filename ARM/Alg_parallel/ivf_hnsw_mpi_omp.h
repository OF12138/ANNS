// =============================================================================
// ivf_hnsw_mpi_omp.h -- IVF+HNSW with MPI (nprobe-split) x OMP (multi-entry)
//
// MPI level: nprobe clusters are split across P processes.
//   Process r searches probe_ids[ lo_p .. hi_p ) where:
//     lo_p = r*(np/P) + min(r, np%P)
//     hi_p = lo_p + (np/P) + (r < np%P ? 1 : 0)
//   Each process has the full IVFHNSWIndex; coarse scan runs independently
//   on every process (same result, no broadcast needed).
//   MPI_Gather collects k candidates from every process at rank 0, which
//   merges them into the global top-k.
//
// OMP level: within each assigned cluster's HNSW, T OMP threads each run an
//   independent beam search from a different entry point (multi-entry strategy
//   from hnsw_simd_parallel.h).  The T results are merged per-cluster before
//   contributing to the local top-k heap.
//
// Platform: AArch64.  Requires MPI + OpenMP.
// =============================================================================
#pragma once

#include <mpi.h>
#include <cfloat>
#include <set>
#include <vector>
#include <queue>
#include <utility>
#include <algorithm>
#include <cstdint>

#include "ivf_hnsw.h"              // IVFHNSWIndex
#include "hnsw_simd_parallel.h"    // hnsw_search_multi_entry_omp
#include "flat_simd.h"             // simd_inner_product_neon_unroll


// =============================================================================
// IVFHNSWMPITimings -- per-query phase timers (accumulated over the batch)
// =============================================================================
struct IVFHNSWMPITimings {
    double t_coarse_s = 0.0;   // centroid scan + partial_sort
    double t_fine_s   = 0.0;   // HNSW fine search (this process's cluster slice)
};


// =============================================================================
// ivf_hnsw_mpi_omp_search_query
//
// All P processes collaborate on a single query:
//   1. Each process runs the coarse scan independently (deterministic, no comm).
//   2. Each process searches its nprobe/P cluster slice via OMP multi-entry HNSW.
//   3. MPI_Gather collects k candidates from every process at rank 0.
//   4. Rank 0 merges P*k candidates into the global top-k heap.
//
// Parameters:
//   idx      : full IVFHNSWIndex (all processes hold the complete index)
//   query    : query vector [vecdim]
//   k        : number of results
//   nprobe   : total clusters to search across all P processes
//   ef       : HNSW beam width per OMP thread
//   rank     : MPI rank of this process
//   size     : total MPI processes
//   nthreads : OMP threads per process (clamped to cluster size if smaller)
//   tm       : optional timing accumulator (nullptr to skip)
//
// Return: merged top-k heap at rank 0; empty at other ranks.
// =============================================================================
std::priority_queue<std::pair<float, uint32_t>>
ivf_hnsw_mpi_omp_search_query(
    const IVFHNSWIndex& idx,
    const float*        query,
    size_t k, size_t nprobe, size_t ef,
    int rank, int size, int nthreads,
    IVFHNSWMPITimings* tm = nullptr)
{
    const size_t d     = idx.ivf.vecdim;
    const size_t nlist = idx.ivf.nlist;
    const size_t np    = std::min(nprobe, nlist);

    double t0, t1, t2;

    // ── Phase 1: coarse centroid scan (all processes, independent) ────────────
    t0 = MPI_Wtime();
    std::vector<std::pair<float, uint32_t>> coarse(nlist);
    for (size_t c = 0; c < nlist; ++c) {
        float ip = simd_inner_product_neon_unroll(
            idx.ivf.centroids.data() + c * d, query, d);
        coarse[c] = {1.0f - ip, static_cast<uint32_t>(c)};
    }
    std::partial_sort(coarse.begin(), coarse.begin() + np, coarse.end());
    t1 = MPI_Wtime();
    if (tm) tm->t_coarse_s += t1 - t0;

    // ── Assign cluster slice to this process ──────────────────────────────────
    int np_int = static_cast<int>(np);
    int chunk  = np_int / size;
    int rem    = np_int % size;
    int lo_p   = rank * chunk + std::min(rank, rem);
    int hi_p   = lo_p + chunk + (rank < rem ? 1 : 0);

    // ── Phase 2: OMP multi-entry HNSW fine search (this process's slice) ──────
    std::priority_queue<std::pair<float, uint32_t>> local_heap;

    for (int pi = lo_p; pi < hi_p; ++pi) {
        uint32_t c    = coarse[pi].second;
        auto*    hnsw = idx.clusters[c];
        if (!hnsw) continue;

        size_t csz = idx.ivf.invlists[c].size();
        // Clamp OMP threads to cluster size (avoid out-of-range entry points)
        int t_eff = std::min(nthreads, static_cast<int>(csz));

        auto local_res = hnsw_search_multi_entry_omp(
            hnsw, query, k, ef, csz, t_eff);

        // Deduplicate IDs from multi-entry results before merging:
        // T threads explore the same small graph and often return the same
        // vector from different entry points.  Without dedup, duplicate IDs
        // occupy heap slots, leaving fewer than k unique results.
        std::set<uint32_t> seen;
        while (!local_res.empty()) {
            float    dist = local_res.top().first;
            uint32_t orig = local_res.top().second;
            local_res.pop();
            if (!seen.insert(orig).second) continue;
            if (local_heap.size() < k) {
                local_heap.push({dist, orig});
            } else if (dist < local_heap.top().first) {
                local_heap.pop();
                local_heap.push({dist, orig});
            }
        }
    }
    t2 = MPI_Wtime();
    if (tm) tm->t_fine_s += t2 - t1;

    // ── MPI_Gather: send k candidates from each process to rank 0 ─────────────
    // Pack local_heap into fixed-size flat arrays; unused slots get FLT_MAX / -1.
    std::vector<float>    local_dists(k, FLT_MAX);
    std::vector<uint32_t> local_ids  (k, static_cast<uint32_t>(-1));
    {
        auto heap_copy = local_heap;
        int pos = 0;
        while (!heap_copy.empty() && pos < static_cast<int>(k)) {
            local_dists[pos] = heap_copy.top().first;
            local_ids  [pos] = heap_copy.top().second;
            heap_copy.pop();
            ++pos;
        }
    }

    std::vector<float>    all_dists;
    std::vector<uint32_t> all_ids;
    if (rank == 0) {
        all_dists.resize(k * size);
        all_ids  .resize(k * size);
    }
    MPI_Gather(local_dists.data(), static_cast<int>(k), MPI_FLOAT,
               all_dists.data(),   static_cast<int>(k), MPI_FLOAT,
               0, MPI_COMM_WORLD);
    MPI_Gather(local_ids.data(),   static_cast<int>(k), MPI_UNSIGNED,
               all_ids.data(),     static_cast<int>(k), MPI_UNSIGNED,
               0, MPI_COMM_WORLD);

    // ── Rank 0: merge P*k candidates into global top-k ────────────────────────
    std::priority_queue<std::pair<float, uint32_t>> global_heap;
    if (rank == 0) {
        for (int p = 0; p < size; ++p) {
            for (size_t j = 0; j < k; ++j) {
                float    d  = all_dists[p * k + j];
                uint32_t id = all_ids  [p * k + j];
                if (d == FLT_MAX) continue;
                if (global_heap.size() < k) {
                    global_heap.push({d, id});
                } else if (d < global_heap.top().first) {
                    global_heap.pop();
                    global_heap.push({d, id});
                }
            }
        }
    }
    return global_heap;
}
