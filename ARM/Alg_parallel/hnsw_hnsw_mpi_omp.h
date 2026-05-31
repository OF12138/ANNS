// =============================================================================
// hnsw_hnsw_mpi_omp.h -- Two-level HNSW (HNSW on HNSW) with MPI x OMP
//
// Identical structure to ivf_hnsw_mpi_omp.h with one change:
//   Coarse phase uses HNSW graph navigation on centroids instead of flat scan.
//
// MPI level: nprobe selected centroids are split across P processes.
//   Each process searches its slice of the fine cluster HNSWs.
//   MPI_Gather collects k candidates at rank 0 which merges the global top-k.
//
// OMP level: within each assigned cluster's HNSW, T threads run independent
//   beam searches from different entry points (multi-entry, dedup merge).
//
// Platform: AArch64.  Requires MPI + OpenMP.
// =============================================================================
#pragma once

#include <mpi.h>
#include <cfloat>
#include <vector>
#include <queue>
#include <utility>
#include <algorithm>
#include <cstdint>

#include "hnsw_hnsw.h"            // HNSWHNSWIndex
#include "hnsw_simd_parallel.h"   // hnsw_search_multi_entry_omp_v2
#include "hnsw_simd.h"            // hnsw_search_simd (for coarse HNSW)


// =============================================================================
// HNSWHNSWMPITimings -- per-query phase timers accumulated over the batch
// =============================================================================
struct HNSWHNSWMPITimings {
    double t_coarse_s = 0.0;   // coarse HNSW graph navigation
    double t_fine_s   = 0.0;   // fine HNSW search across assigned clusters
};


// =============================================================================
// hnsw_hnsw_mpi_omp_search_query
//
// All P processes collaborate on a single query:
//   1. Each process runs the coarse HNSW search independently (deterministic).
//   2. Each process fine-searches its nprobe/P cluster slice via OMP multi-entry.
//   3. MPI_Gather collects k candidates from every process at rank 0.
//   4. Rank 0 merges P*k candidates into the global top-k heap.
//
// Parameters:
//   idx       : full HNSWHNSWIndex (all processes hold the complete index)
//   query     : query vector [vecdim]
//   k         : number of results
//   nprobe    : total clusters to search across all P processes
//   ef_coarse : beam width for coarse HNSW centroid navigation
//   ef_fine   : HNSW beam width per OMP thread for fine search
//   rank      : MPI rank of this process
//   size      : total MPI processes
//   nthreads  : OMP threads per process
//   tm        : optional timing accumulator (nullptr to skip)
//
// Return: merged top-k heap at rank 0; empty at other ranks.
// =============================================================================
std::priority_queue<std::pair<float, uint32_t>>
hnsw_hnsw_mpi_omp_search_query(
    const HNSWHNSWIndex& idx,
    const float*         query,
    size_t k, size_t nprobe, size_t ef_coarse, size_t ef_fine,
    int rank, int size, int nthreads,
    HNSWHNSWMPITimings* tm = nullptr)
{
    const size_t np = std::min(nprobe, idx.inner.ivf.nlist);

    double t0, t1, t2;

    // ── Phase 1: coarse HNSW navigation (all processes, independent) ──────────
    t0 = MPI_Wtime();

    auto coarse_heap = hnsw_search_simd(idx.coarse_hnsw, query, np, ef_coarse);

    // Convert max-heap → vector sorted by distance ascending (closest first)
    std::vector<std::pair<float, uint32_t>> coarse;
    coarse.reserve(coarse_heap.size());
    while (!coarse_heap.empty()) {
        coarse.push_back(coarse_heap.top());
        coarse_heap.pop();
    }
    std::sort(coarse.begin(), coarse.end());

    t1 = MPI_Wtime();
    if (tm) tm->t_coarse_s += t1 - t0;

    // ── Assign cluster slice to this process ──────────────────────────────────
    int np_int = static_cast<int>(coarse.size());
    int chunk  = np_int / size;
    int rem    = np_int % size;
    int lo_p   = rank * chunk + std::min(rank, rem);
    int hi_p   = lo_p + chunk + (rank < rem ? 1 : 0);

    // ── Phase 2: OMP multi-entry HNSW fine search (this process's slice) ──────
    std::priority_queue<std::pair<float, uint32_t>> local_heap;

    for (int pi = lo_p; pi < hi_p; ++pi) {
        uint32_t c    = coarse[pi].second;
        auto*    hnsw = idx.inner.clusters[c];
        if (!hnsw) continue;

        size_t csz   = idx.inner.ivf.invlists[c].size();
        int    t_eff = std::min(nthreads, static_cast<int>(csz));

        auto local_res = hnsw_search_multi_entry_omp_v2(
            hnsw, query, k, ef_fine, csz, t_eff);

        while (!local_res.empty()) {
            float    dist = local_res.top().first;
            uint32_t orig = local_res.top().second;
            local_res.pop();
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
        all_dists.resize(k * static_cast<size_t>(size));
        all_ids  .resize(k * static_cast<size_t>(size));
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
                float    d  = all_dists[static_cast<size_t>(p) * k + j];
                uint32_t id = all_ids  [static_cast<size_t>(p) * k + j];
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
