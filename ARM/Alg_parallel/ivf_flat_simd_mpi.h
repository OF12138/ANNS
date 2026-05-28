// =============================================================================
// ivf_flat_simd_mpi.h — MPI-parallel IVF-SIMD search (nprobe partition)
//
// Parallelization strategy: split the nprobe selected clusters across processes
//
//   Build:
//     Each MPI process independently loads the same pre-built IVFIndex from a
//     shared NFS cache file.  No index broadcast is needed.
//
//   Per-query search (ivf_mpi_search_query):
//
//     Step 1 — ALL processes: Bcast query vector, then run coarse independently
//       Rank 0 broadcasts the query vector (vecdim floats = 384 B).
//       Every process then runs the full coarse phase against all nlist centroids
//       and derives the top-nprobe cluster list independently.
//
//       Why redundant coarse is correct here:
//         - Centroids are replicated on every process, so the result is identical.
//         - While ranks 1..P-1 wait for the Bcast anyway, they use that time (and
//           a little more) to run coarse in parallel with rank 0 rather than
//           idling after the Bcast waiting for a second cluster-ID Bcast.
//         - Eliminating the cluster-ID Bcast removes one MPI round-trip (~2–5 µs)
//           at the cost of (P-1) redundant coarse phases that run in parallel —
//           wall-clock time for coarse stays one coarse-phase worth regardless.
//         - Net effect: same wall time, one fewer MPI call, simpler code.
//
//     Step 2 — ALL processes: fine scan over assigned probe slice
//       Process r scans clusters at probe positions [lo_p, hi_p) in the sorted
//       top-nprobe list, where:
//           lo_p = r       * nprobe / size
//           hi_p = (r + 1) * nprobe / size
//       This guarantees every process handles exactly nprobe/P (±1) clusters,
//       regardless of where in the ID space those clusters fall.
//       Remaining load imbalance comes only from cluster SIZE variance.
//
//     Step 3 — MPI_Gather: each process sends its local top-k to rank 0
//       Fixed-size buffer: k (dist, id) pairs; pad with (FLT_MAX, UINT32_MAX).
//
//     Step 4 — RANK 0: merge P × k candidates → global top-k
//
//   Why nprobe split is better than nlist-range split:
//     With nlist-range split, ALL of the top-nprobe clusters might fall in one
//     process's range while others do zero work.  With nprobe-list split, every
//     process always gets exactly floor(nprobe/P) or ceil(nprobe/P) clusters.
//
//   Communication per query:
//     Bcast:  vecdim × 4 bytes              ≈ 384 B  (query only)
//     Gather: 2 × size × k × 4 bytes        ≈ 640 B  (size=8, k=10)
//     Both are small; on-node MPI latency (~2–5 µs) dominates.
//
// Timing:
//   ivf_mpi_search_query_timed adds six MPI_Wtime() calls per query (~50 ns
//   each on most implementations).  Total overhead ≈ 300 ns/query — negligible
//   against µs-scale operations and invisible in the latency numbers.
//   Use this version only in the measured run; the warm-up uses the plain version.
//
// Compile:  mpicxx main_mpi.cc -o main_mpi -O2 -std=c++11 -lm
// Run:      mpiexec -n 4 ./main_mpi
// Platform: AArch64, OpenMPI / MPICH >= 2.2
// =============================================================================
#pragma once
#include <mpi.h>
#include <vector>
#include <queue>
#include <utility>
#include <algorithm>
#include <cfloat>
#include <cstdint>
#include "ivf_flat_simd.h"   // IVFIndex, simd_inner_product_neon_unroll


// =============================================================================
// ivf_mpi_search_query — MPI-collective IVF-SIMD search for one query
//
// MUST be called collectively by ALL MPI processes.
//
// INPUT (all processes):
//   idx     — full IVFIndex loaded independently on every process
//   base    — full base dataset [base_number × vecdim]
//   query   — query vector [vecdim], valid on rank 0; ignored on other ranks
//              before the broadcast
//   k       — top-k results to return
//   nprobe  — number of clusters to probe (latency-recall knob); must be
//              divisible by size for perfectly even partition (remainder goes
//              to the last rank)
//   rank    — MPI rank of this process
//   size    — total MPI process count
//
// OUTPUT:
//   rank 0  — max-heap of (1.0-IP distance, vector_id) pairs, size k
//   rank >0 — empty heap
// =============================================================================
std::priority_queue<std::pair<float, uint32_t>>
ivf_mpi_search_query(
    const IVFIndex& idx,
    const float*    base,
    const float*    query,
    size_t k, size_t nprobe,
    int rank, int size)
{
    const size_t d     = idx.vecdim;
    const size_t nlist = idx.nlist;
    const size_t np    = std::min(nprobe, nlist);

    // ── Step 1: Broadcast query, then all processes run coarse independently ───
    // Rank 0 broadcasts the query vector; all processes (including rank 0) then
    // run the identical coarse phase locally.  No cluster-ID Bcast needed.
    std::vector<float> q_buf(d);
    if (rank == 0)
        std::copy(query, query + d, q_buf.begin());
    MPI_Bcast(q_buf.data(), static_cast<int>(d), MPI_FLOAT, 0, MPI_COMM_WORLD);
    const float* q = q_buf.data();

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

    // ── Step 2: Fine scan — each process handles its slice of probe_ids ───────
    // Process r scans probe positions [lo_p, hi_p) in the sorted top-nprobe list.
    // This guarantees every process scans exactly nprobe/P (±1) clusters,
    // regardless of where in the ID space those clusters fall.
    const size_t lo_p = static_cast<size_t>(rank)     * np / static_cast<size_t>(size);
    const size_t hi_p = static_cast<size_t>(rank + 1) * np / static_cast<size_t>(size);

    std::priority_queue<std::pair<float, uint32_t>> local_heap;

    for (size_t probe = lo_p; probe < hi_p; ++probe) {
        const uint32_t c = probe_ids[probe];
        for (uint32_t orig : idx.invlists[c]) {
            float ip  = simd_inner_product_neon_unroll(
                base + static_cast<size_t>(orig) * d, q, d);
            float dis = 1.0f - ip;
            if (local_heap.size() < k) {
                local_heap.push({dis, orig});
            } else if (dis < local_heap.top().first) {
                local_heap.pop();
                local_heap.push({dis, orig});
            }
        }
    }

    // ── Step 3: Pack local heap into fixed-size flat arrays ───────────────────
    // Pad unused slots with sentinel values so MPI_Gather can use fixed counts.
    std::vector<float>    send_dist(k, FLT_MAX);
    std::vector<uint32_t> send_ids (k, UINT32_MAX);
    for (size_t i = 0; i < k && !local_heap.empty(); ++i) {
        send_dist[i] = local_heap.top().first;
        send_ids [i] = local_heap.top().second;
        local_heap.pop();
    }

    // ── Step 4: Gather all local top-k to rank 0 ─────────────────────────────
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

    // ── Step 5: Rank 0 merges P × k candidates → global top-k ────────────────
    std::priority_queue<std::pair<float, uint32_t>> result;
    if (rank == 0) {
        const size_t total = static_cast<size_t>(size) * k;
        for (size_t i = 0; i < total; ++i) {
            if (all_ids[i] == UINT32_MAX) continue;   // padding entry
            const float    dis = all_dist[i];
            const uint32_t vid = all_ids [i];
            if (result.size() < k) {
                result.push({dis, vid});
            } else if (dis < result.top().first) {
                result.pop();
                result.push({dis, vid});
            }
        }
    }
    return result;
}


// =============================================================================
// IVFMPITimings — per-query phase timing accumulators (seconds)
//
// All fields are accumulated over multiple queries; divide by query count for
// per-query averages.  Fields are local to each process except t_merge which
// is only meaningful on rank 0.
//
//   t_bcast_s   : wall time inside MPI_Bcast (query vector)
//   t_coarse_s  : local coarse phase (centroid IP + partial_sort)
//   t_fine_s    : local fine scan (vector IP over owned clusters)
//   t_gather_s  : wall time inside both MPI_Gather calls
//   t_merge_s   : rank 0 merge of P×k candidates (rank 0 only; 0 on others)
// =============================================================================
struct IVFMPITimings {
    double t_bcast_s  = 0.0;
    double t_coarse_s = 0.0;
    double t_fine_s   = 0.0;
    double t_gather_s = 0.0;
    double t_merge_s  = 0.0;
};


// =============================================================================
// ivf_mpi_search_query_timed — same as ivf_mpi_search_query with phase timers
//
// Accumulates wall-clock seconds (via MPI_Wtime) into *tm on every call.
// Overhead: 6 × MPI_Wtime() calls ≈ 300 ns/query — negligible.
//
// INPUT / OUTPUT: same as ivf_mpi_search_query, plus:
//   tm  — pointer to IVFMPITimings; fields are *added to* on each call.
//         Caller must zero-initialise before the first call.
// =============================================================================
std::priority_queue<std::pair<float, uint32_t>>
ivf_mpi_search_query_timed(
    const IVFIndex& idx,
    const float*    base,
    const float*    query,
    size_t k, size_t nprobe,
    int rank, int size,
    IVFMPITimings* tm)
{
    const size_t d     = idx.vecdim;
    const size_t nlist = idx.nlist;
    const size_t np    = std::min(nprobe, nlist);

    // ── Step 1: Bcast query, then all processes run coarse independently ───────
    std::vector<float> q_buf(d);
    if (rank == 0)
        std::copy(query, query + d, q_buf.begin());

    double t0 = MPI_Wtime();
    MPI_Bcast(q_buf.data(), static_cast<int>(d), MPI_FLOAT, 0, MPI_COMM_WORLD);
    tm->t_bcast_s += MPI_Wtime() - t0;

    const float* q = q_buf.data();

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

    // ── Step 2: Fine scan ─────────────────────────────────────────────────────
    const size_t lo_p = static_cast<size_t>(rank)     * np / static_cast<size_t>(size);
    const size_t hi_p = static_cast<size_t>(rank + 1) * np / static_cast<size_t>(size);

    std::priority_queue<std::pair<float, uint32_t>> local_heap;

    t0 = MPI_Wtime();
    for (size_t probe = lo_p; probe < hi_p; ++probe) {
        const uint32_t c = probe_ids[probe];
        for (uint32_t orig : idx.invlists[c]) {
            float ip  = simd_inner_product_neon_unroll(
                base + static_cast<size_t>(orig) * d, q, d);
            float dis = 1.0f - ip;
            if (local_heap.size() < k) {
                local_heap.push({dis, orig});
            } else if (dis < local_heap.top().first) {
                local_heap.pop();
                local_heap.push({dis, orig});
            }
        }
    }
    tm->t_fine_s += MPI_Wtime() - t0;

    // ── Step 3: Pack + Gather ─────────────────────────────────────────────────
    std::vector<float>    send_dist(k, FLT_MAX);
    std::vector<uint32_t> send_ids (k, UINT32_MAX);
    for (size_t i = 0; i < k && !local_heap.empty(); ++i) {
        send_dist[i] = local_heap.top().first;
        send_ids [i] = local_heap.top().second;
        local_heap.pop();
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

    // ── Step 4: Merge (rank 0 only) ───────────────────────────────────────────
    std::priority_queue<std::pair<float, uint32_t>> result;
    if (rank == 0) {
        t0 = MPI_Wtime();
        const size_t total = static_cast<size_t>(size) * k;
        for (size_t i = 0; i < total; ++i) {
            if (all_ids[i] == UINT32_MAX) continue;
            const float    dis = all_dist[i];
            const uint32_t vid = all_ids [i];
            if (result.size() < k) {
                result.push({dis, vid});
            } else if (dis < result.top().first) {
                result.pop();
                result.push({dis, vid});
            }
        }
        tm->t_merge_s += MPI_Wtime() - t0;
    }
    return result;
}
