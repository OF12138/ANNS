// =============================================================================
// hnsw_partition_mpi_omp.h -- Partitioned HNSW with MPI x OMP search
//
// MPI level: num_parts partitions are split across P processes.
//   Process r searches partitions [lo_p .. hi_p) where:
//     lo_p = r*(num_parts/P) + min(r, num_parts%P)
//     hi_p = lo_p + (num_parts/P) + (r < num_parts%P ? 1 : 0)
//   Each process holds ONLY its assigned partition HNSWs in memory.
//   MPI_Gather collects k candidates from every process at rank 0, which
//   merges them into the global top-k.
//
// OMP level: within each assigned partition's HNSW, T threads each run an
//   independent beam search from a different entry point
//   (hnsw_search_multi_entry_omp_v2 with dedup merge).
//
// Unlike IVF+HNSW, there is no coarse scan: every partition is always searched
// (by some process), so no candidate is ever skipped by quantization error.
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

#include "hnsw_partition.h"       // PartitionedHNSWIndex
#include "hnsw_simd_parallel.h"   // hnsw_search_multi_entry_omp_v2


// =============================================================================
// HNSWPartitionTimings -- per-query phase timer (accumulated over the batch)
// =============================================================================
struct HNSWPartitionTimings {
    double t_search_s = 0.0;   // HNSW fine search across assigned partitions
};


// =============================================================================
// hnsw_partition_mpi_omp_search
//
// All P processes collaborate on a single query:
//   1. Each process searches its partition slice via OMP multi-entry HNSW.
//   2. MPI_Gather collects k candidates from every process at rank 0.
//   3. Rank 0 merges P*k candidates into the global top-k heap.
//
// Parameters:
//   idx      : PartitionedHNSWIndex (each process holds only its partitions)
//   query    : query vector [vecdim]
//   k        : number of results
//   ef       : HNSW beam width per OMP thread
//   rank     : MPI rank of this process
//   size     : total MPI processes
//   nthreads : OMP threads per process
//   tm       : optional timing accumulator (nullptr to skip)
//
// Return: merged top-k heap at rank 0; empty at other ranks.
// =============================================================================
std::priority_queue<std::pair<float, uint32_t>>
hnsw_partition_mpi_omp_search(
    const PartitionedHNSWIndex& idx,
    const float*                query,
    size_t k, size_t ef,
    int rank, int size, int nthreads,
    HNSWPartitionTimings* tm = nullptr)
{
    const int np_int = static_cast<int>(idx.num_parts);
    const int chunk  = np_int / size;
    const int rem    = np_int % size;
    const int lo_p   = rank * chunk + std::min(rank, rem);
    const int hi_p   = lo_p + chunk + (rank < rem ? 1 : 0);

    double t0 = MPI_Wtime();

    // Search assigned partitions with OMP multi-entry beam search
    std::priority_queue<std::pair<float, uint32_t>> local_heap;

    for (int p = lo_p; p < hi_p; ++p) {
        auto* hnsw = idx.parts[static_cast<size_t>(p)];
        if (!hnsw) continue;

        // Use actual node count for entry-point spread (handles last-partition edge case)
        size_t psz  = static_cast<size_t>(hnsw->cur_element_count);
        int    t_eff = std::min(nthreads, static_cast<int>(psz));

        auto local_res = hnsw_search_multi_entry_omp_v2(hnsw, query, k, ef, psz, t_eff);

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

    double t1 = MPI_Wtime();
    if (tm) tm->t_search_s += t1 - t0;

    // Pack local top-k for MPI_Gather; unused slots padded with FLT_MAX / -1
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

    // Rank 0: merge P*k candidates into global top-k
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
