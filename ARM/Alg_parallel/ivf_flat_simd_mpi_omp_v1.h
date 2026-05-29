// =============================================================================
// ivf_flat_simd_mpi_omp_v1.h — Plan I hybrid: nprobe-MPI × OMP fine scan
//
// Parallelization:
//   MPI layer : split nprobe selected clusters across P processes (same as
//               pure-MPI ivf_flat_simd_mpi.h)
//   OMP layer : within each process, T threads scan the assigned cluster
//               slice in parallel (schedule dynamic for load balance)
//
// Per-query steps (ALL processes collectively):
//   1. [MPI_Bcast]  rank 0 → all:  query vector
//   2. Coarse       all processes independently rank all nlist centroids
//   3. Fine         process r owns probe_ids[lo_p..hi_p);
//                   T OMP threads share that slice via dynamic scheduling
//   4. [MPI_Gather] each process → rank 0:  local top-k (k pairs)
//   5. Merge        rank 0 merges P×k candidates → global top-k
//
// Compile:  mpicxx ... -fopenmp
// =============================================================================
#pragma once
#include <mpi.h>
#include <omp.h>
#include <vector>
#include <queue>
#include <utility>
#include <algorithm>
#include <cfloat>
#include <cstdint>
#include "ivf_flat_simd.h"


struct MPIV1Timings {
    double t_bcast_s  = 0.0;
    double t_coarse_s = 0.0;
    double t_fine_s   = 0.0;   // wall time of OMP parallel region
    double t_gather_s = 0.0;
    double t_merge_s  = 0.0;
};


std::priority_queue<std::pair<float, uint32_t>>
ivf_mpi_omp_v1_search_query(
    const IVFIndex& idx,
    const float*    base,
    const float*    query,
    size_t k, size_t nprobe,
    int rank, int size, int nthreads)
{
    const size_t d     = idx.vecdim;
    const size_t nlist = idx.nlist;
    const size_t np    = std::min(nprobe, nlist);

    // ── Step 1: Bcast query ───────────────────────────────────────────────────
    std::vector<float> q_buf(d);
    if (rank == 0) std::copy(query, query + d, q_buf.begin());
    MPI_Bcast(q_buf.data(), static_cast<int>(d), MPI_FLOAT, 0, MPI_COMM_WORLD);
    const float* q = q_buf.data();

    // ── Step 2: Coarse (all processes, independently) ─────────────────────────
    std::vector<uint32_t> probe_ids(np);
    {
        std::vector<std::pair<float, uint32_t>> coarse(nlist);
        for (size_t c = 0; c < nlist; ++c) {
            float ip = simd_inner_product_neon_unroll(
                idx.centroids.data() + c * d, q, d);
            coarse[c] = {1.0f - ip, static_cast<uint32_t>(c)};
        }
        std::partial_sort(coarse.begin(), coarse.begin() + np, coarse.end());
        for (size_t i = 0; i < np; ++i)
            probe_ids[i] = coarse[i].second;
    }

    // ── Step 3: Fine scan (nprobe slice × OMP threads) ────────────────────────
    const size_t lo_p = static_cast<size_t>(rank)     * np / static_cast<size_t>(size);
    const size_t hi_p = static_cast<size_t>(rank + 1) * np / static_cast<size_t>(size);

    std::vector<std::priority_queue<std::pair<float, uint32_t>>> thread_heaps(nthreads);

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        auto& heap = thread_heaps[tid];

        #pragma omp for schedule(dynamic, 1)
        for (int probe = static_cast<int>(lo_p);
                 probe < static_cast<int>(hi_p); ++probe) {
            const uint32_t c = probe_ids[probe];
            for (uint32_t orig : idx.invlists[c]) {
                float ip  = simd_inner_product_neon_unroll(
                    base + static_cast<size_t>(orig) * d, q, d);
                float dis = 1.0f - ip;
                if (heap.size() < k)
                    heap.push({dis, orig});
                else if (dis < heap.top().first) {
                    heap.pop(); heap.push({dis, orig});
                }
            }
        }
    }

    // Merge thread heaps → process local top-k
    std::priority_queue<std::pair<float, uint32_t>> local_top;
    for (auto& h : thread_heaps) {
        while (!h.empty()) {
            auto top = h.top(); h.pop();
            if (local_top.size() < k) local_top.push(top);
            else if (top.first < local_top.top().first) {
                local_top.pop(); local_top.push(top);
            }
        }
    }

    // ── Step 4: Pack and Gather ───────────────────────────────────────────────
    std::vector<float>    send_dist(k, FLT_MAX);
    std::vector<uint32_t> send_ids (k, UINT32_MAX);
    for (size_t i = 0; i < k && !local_top.empty(); ++i) {
        send_dist[i] = local_top.top().first;
        send_ids [i] = local_top.top().second;
        local_top.pop();
    }

    std::vector<float>    all_dist;
    std::vector<uint32_t> all_ids;
    if (rank == 0) {
        all_dist.resize(static_cast<size_t>(size) * k);
        all_ids .resize(static_cast<size_t>(size) * k);
    }

    MPI_Gather(send_dist.data(), static_cast<int>(k), MPI_FLOAT,
               all_dist.data(), static_cast<int>(k), MPI_FLOAT,
               0, MPI_COMM_WORLD);
    MPI_Gather(send_ids.data(),  static_cast<int>(k), MPI_UNSIGNED,
               all_ids.data(),   static_cast<int>(k), MPI_UNSIGNED,
               0, MPI_COMM_WORLD);

    // ── Step 5: Merge (rank 0) ────────────────────────────────────────────────
    std::priority_queue<std::pair<float, uint32_t>> result;
    if (rank == 0) {
        const size_t total = static_cast<size_t>(size) * k;
        for (size_t i = 0; i < total; ++i) {
            if (all_ids[i] == UINT32_MAX) continue;
            const float dis    = all_dist[i];
            const uint32_t vid = all_ids [i];
            if (result.size() < k) result.push({dis, vid});
            else if (dis < result.top().first) {
                result.pop(); result.push({dis, vid});
            }
        }
    }
    return result;
}


std::priority_queue<std::pair<float, uint32_t>>
ivf_mpi_omp_v1_search_query_timed(
    const IVFIndex& idx,
    const float*    base,
    const float*    query,
    size_t k, size_t nprobe,
    int rank, int size, int nthreads,
    MPIV1Timings* tm)
{
    const size_t d     = idx.vecdim;
    const size_t nlist = idx.nlist;
    const size_t np    = std::min(nprobe, nlist);

    // Bcast
    std::vector<float> q_buf(d);
    if (rank == 0) std::copy(query, query + d, q_buf.begin());
    double t0 = MPI_Wtime();
    MPI_Bcast(q_buf.data(), static_cast<int>(d), MPI_FLOAT, 0, MPI_COMM_WORLD);
    tm->t_bcast_s += MPI_Wtime() - t0;
    const float* q = q_buf.data();

    // Coarse
    std::vector<uint32_t> probe_ids(np);
    t0 = MPI_Wtime();
    {
        std::vector<std::pair<float, uint32_t>> coarse(nlist);
        for (size_t c = 0; c < nlist; ++c) {
            float ip = simd_inner_product_neon_unroll(
                idx.centroids.data() + c * d, q, d);
            coarse[c] = {1.0f - ip, static_cast<uint32_t>(c)};
        }
        std::partial_sort(coarse.begin(), coarse.begin() + np, coarse.end());
        for (size_t i = 0; i < np; ++i)
            probe_ids[i] = coarse[i].second;
    }
    tm->t_coarse_s += MPI_Wtime() - t0;

    // Fine scan (OMP)
    const size_t lo_p = static_cast<size_t>(rank)     * np / static_cast<size_t>(size);
    const size_t hi_p = static_cast<size_t>(rank + 1) * np / static_cast<size_t>(size);

    std::vector<std::priority_queue<std::pair<float, uint32_t>>> thread_heaps(nthreads);

    t0 = MPI_Wtime();
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        auto& heap = thread_heaps[tid];
        #pragma omp for schedule(dynamic, 1)
        for (int probe = static_cast<int>(lo_p);
                 probe < static_cast<int>(hi_p); ++probe) {
            const uint32_t c = probe_ids[probe];
            for (uint32_t orig : idx.invlists[c]) {
                float ip  = simd_inner_product_neon_unroll(
                    base + static_cast<size_t>(orig) * d, q, d);
                float dis = 1.0f - ip;
                if (heap.size() < k) heap.push({dis, orig});
                else if (dis < heap.top().first) { heap.pop(); heap.push({dis, orig}); }
            }
        }
    }
    tm->t_fine_s += MPI_Wtime() - t0;

    // Merge thread heaps → local top-k
    std::priority_queue<std::pair<float, uint32_t>> local_top;
    for (auto& h : thread_heaps) {
        while (!h.empty()) {
            auto top = h.top(); h.pop();
            if (local_top.size() < k) local_top.push(top);
            else if (top.first < local_top.top().first) { local_top.pop(); local_top.push(top); }
        }
    }

    // Pack + Gather
    std::vector<float>    send_dist(k, FLT_MAX);
    std::vector<uint32_t> send_ids (k, UINT32_MAX);
    for (size_t i = 0; i < k && !local_top.empty(); ++i) {
        send_dist[i] = local_top.top().first;
        send_ids [i] = local_top.top().second;
        local_top.pop();
    }

    std::vector<float>    all_dist;
    std::vector<uint32_t> all_ids;
    if (rank == 0) {
        all_dist.resize(static_cast<size_t>(size) * k);
        all_ids .resize(static_cast<size_t>(size) * k);
    }

    t0 = MPI_Wtime();
    MPI_Gather(send_dist.data(), static_cast<int>(k), MPI_FLOAT,
               all_dist.data(), static_cast<int>(k), MPI_FLOAT,
               0, MPI_COMM_WORLD);
    MPI_Gather(send_ids.data(),  static_cast<int>(k), MPI_UNSIGNED,
               all_ids.data(),   static_cast<int>(k), MPI_UNSIGNED,
               0, MPI_COMM_WORLD);
    tm->t_gather_s += MPI_Wtime() - t0;

    // Merge (rank 0)
    std::priority_queue<std::pair<float, uint32_t>> result;
    if (rank == 0) {
        t0 = MPI_Wtime();
        const size_t total = static_cast<size_t>(size) * k;
        for (size_t i = 0; i < total; ++i) {
            if (all_ids[i] == UINT32_MAX) continue;
            if (result.size() < k) result.push({all_dist[i], all_ids[i]});
            else if (all_dist[i] < result.top().first) {
                result.pop(); result.push({all_dist[i], all_ids[i]});
            }
        }
        tm->t_merge_s += MPI_Wtime() - t0;
    }
    return result;
}
