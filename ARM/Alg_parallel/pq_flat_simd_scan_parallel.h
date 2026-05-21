// =============================================================================
// pq_flat_simd_scan_parallel.h — Query-parallel coarse scan + rerank for PQ
//
// Parallelizes the ADC scan+rerank phase across a BATCH of queries.
// The LUT (dtable) for each query is assumed to be pre-built and stored in
// all_dtables[i * M*K .. (i+1)*M*K).  Each thread handles a disjoint slice
// of queries, calling pq_rerank_from_dtable_gather for its assigned range.
//
// ── Why this phase is different from LUT ─────────────────────────────────────
// LUT build (pq_build_lut_cc_unroll) is compute-bound: dense SIMD FMAs with
// predictable memory access — scales cleanly with thread count.
//
// Coarse scan (pq_rerank_from_dtable_gather) is memory-bound: for each of
// 100K base vectors, it issues M=8 random-address LUT lookups (gather loads)
// into a 2 KB table.  Multiple threads reading different parts of base[] and
// different LUT rows simultaneously stress the memory bus and L2/L3 bandwidth.
// Expect sub-linear speedup or even negative speedup at high thread counts.
// This is the key analysis point the manual asks you to discuss in the report.
//
// Two variants (Pthread / OpenMP) — identical work distribution, different
// thread management overhead:
//   pq_batch_scan_rerank_pthread — threads created+destroyed once per batch
//   pq_batch_scan_rerank_omp    — OMP persistent pool, just woken per batch
//
// Platform: AArch64. Compile: g++ main.cc -o main -O2 -fopenmp -lpthread -std=c++11
// =============================================================================
#pragma once
#include <pthread.h>
#include <omp.h>
#include <queue>
#include <utility>
#include <vector>
#include <algorithm>
#include <cstdint>
#include "pq_flat_simd.h"   // pq_rerank_from_dtable_gather, PQIndex


// =============================================================================
// Pthread version
// =============================================================================

struct _PQScanArgs {
    const PQIndex* pq_index;
    const float*   base;
    const float*   queries;
    size_t         vecdim;
    size_t         k;
    size_t         p;
    const float*   all_dtables;   // [num_queries × M × K], read-only
    size_t         lut_size;      // M * K
    int            q_start;
    int            q_end;
    std::priority_queue<std::pair<float, uint32_t>>* results;
};

// Worker: runs gather scan+rerank for each query in [q_start, q_end).
static void* _pq_scan_worker(void* arg)
{
    auto* a = static_cast<_PQScanArgs*>(arg);
    for (int i = a->q_start; i < a->q_end; ++i) {
        a->results[i] = pq_rerank_from_dtable_gather(
            *a->pq_index,
            a->base,
            a->queries + (size_t)i * a->vecdim,
            a->k, a->p,
            a->all_dtables + (size_t)i * a->lut_size);
    }
    return nullptr;
}

// pq_batch_scan_rerank_pthread — parallel gather scan+rerank, Pthread
//
// Parameters:
//   pq_index    : PQ index (codes, centroids, dims) — read-only, shared
//   base        : base dataset [base_number × vecdim] — read-only, shared
//   queries     : query batch [num_queries × vecdim]
//   num_queries : number of queries
//   vecdim      : vector dimension
//   k, p        : top-k final results, top-p coarse candidates
//   all_dtables : pre-built LUTs [num_queries × M × K] — read-only, shared
//   results     : caller-allocated output array [num_queries]
//   num_threads : Pthread worker count
//
// Thread lifetime: created once, joined once per batch call.
void pq_batch_scan_rerank_pthread(
    const PQIndex& pq_index,
    const float* base,
    const float* queries, int num_queries, size_t vecdim,
    size_t k, size_t p,
    const float* all_dtables,
    std::priority_queue<std::pair<float, uint32_t>>* results,
    int num_threads)
{
    const size_t lut_size = pq_index.M * pq_index.K;
    std::vector<pthread_t>   tids(num_threads);
    std::vector<_PQScanArgs> args(num_threads);

    int chunk = (num_queries + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        args[t].pq_index    = &pq_index;
        args[t].base        = base;
        args[t].queries     = queries;
        args[t].vecdim      = vecdim;
        args[t].k           = k;
        args[t].p           = p;
        args[t].all_dtables = all_dtables;
        args[t].lut_size    = lut_size;
        args[t].q_start     = t * chunk;
        args[t].q_end       = std::min(num_queries, (t + 1) * chunk);
        args[t].results     = results;
        pthread_create(&tids[t], nullptr, _pq_scan_worker, &args[t]);
    }
    for (int t = 0; t < num_threads; ++t) pthread_join(tids[t], nullptr);
}


// =============================================================================
// OpenMP version
// =============================================================================

// pq_batch_scan_rerank_omp — parallel gather scan+rerank, OpenMP
//
// schedule(static): each query scans the same 100K base vectors — uniform work.
// OMP thread pool is persistent — no per-call spawn cost.
void pq_batch_scan_rerank_omp(
    const PQIndex& pq_index,
    const float* base,
    const float* queries, int num_queries, size_t vecdim,
    size_t k, size_t p,
    const float* all_dtables,
    std::priority_queue<std::pair<float, uint32_t>>* results,
    int num_threads)
{
    const size_t lut_size = pq_index.M * pq_index.K;
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int i = 0; i < num_queries; ++i) 
    {
        results[i] = pq_rerank_from_dtable_gather(
            pq_index,
            base,
            queries + (size_t)i * vecdim,
            k, p,
            all_dtables + (size_t)i * lut_size);
    }
}
