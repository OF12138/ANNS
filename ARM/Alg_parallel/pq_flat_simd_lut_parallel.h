// =============================================================================
// pq_flat_simd_lut_parallel.h — Query-parallel LUT construction for PQ-SIMD
//
// Parallelizes the LUT build phase across a BATCH of queries using query-level
// parallelism. Inner LUT kernel: pq_build_lut_cc_unroll (best single-thread
// LUT from pq_flat_simd.h — cross-centroid + 4× unroll, 0 reductions).
//
// ── Strategy: Query-level parallelism ────────────────────────────────────────
// The LUT build for different queries is embarrassingly parallel:
//   query i needs only test_query[i] and the shared (read-only) PQIndexSIMD.
//   Each thread writes to all_dtables[i * M*K .. (i+1)*M*K) — disjoint ranges.
//   No locks or barriers needed during compute.
//
// Two variants:
//   pq_batch_build_lut_pthread — static Pthread pool: threads created once,
//     joined once, each processes ceil(num_queries/num_threads) queries.
//   pq_batch_build_lut_omp — OMP persistent pool:
//     #pragma omp parallel for schedule(static) over query index.
//
// Usage in experiments:
//   1. Call pq_batch_build_lut_pthread / _omp around gettimeofday to measure
//      parallel LUT time. Compare against [phase] avg LUT build from SEARCH_ALG 13.
//   2. Pass the pre-built all_dtables to pq_rerank_from_dtable_gather for the
//      coarse scan + rerank (unchanged, single-thread per query in the loop).
//
// Platform: AArch64. Compile: g++ main.cc -o main -O2 -fopenmp -lpthread -std=c++11
// =============================================================================
#pragma once
#include <pthread.h>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <cstdint>
#include "pq_flat_simd.h"   // pq_build_lut_cc_unroll, PQIndexSIMD


// =============================================================================
// Pthread version
// =============================================================================

struct _PQLUTArgs {
    const PQIndexSIMD* pq_simd;
    const float*       queries;
    size_t             vecdim;
    size_t             lut_size;   // M * K floats per query
    int                q_start;   // inclusive
    int                q_end;     // exclusive
    float*             all_dtables;
};

// Worker: builds LUT for each query in [q_start, q_end) using cc_unroll kernel.
static void* _pq_lut_worker(void* arg)
{
    auto* a = static_cast<_PQLUTArgs*>(arg);
    for (int i = a->q_start; i < a->q_end; ++i) 
    {
        pq_build_lut_cc_unroll(
            *a->pq_simd,
            a->queries    + (size_t)i * a->vecdim,
            a->all_dtables + (size_t)i * a->lut_size);
    }
    return nullptr;
}

// pq_batch_build_lut_pthread — LUT build for all num_queries queries, Pthread
//
// Parameters:
//   pq_simd     : transposed centroid layout (read-only, shared across threads)
//   queries     : all query vectors [num_queries × vecdim]
//   num_queries : number of queries to process
//   vecdim      : query vector dimension
//   all_dtables : caller-allocated output [num_queries × M × K]
//   num_threads : number of Pthread worker threads to spawn
//
// Each thread owns queries [q_start, q_end); writes to disjoint dtable slots.
// Thread lifetime: created once at call entry, joined once at call exit.
void pq_batch_build_lut_pthread(
    const PQIndexSIMD& pq_simd,
    const float* queries, int num_queries, size_t vecdim,
    float* all_dtables,
    int num_threads)
{
    const size_t lut_size = pq_simd.idx->M * pq_simd.idx->K;
    std::vector<pthread_t>  tids(num_threads);
    std::vector<_PQLUTArgs> args(num_threads);

    int chunk = (num_queries + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; ++t) 
    {
        args[t].pq_simd     = &pq_simd;
        args[t].queries     = queries;
        args[t].vecdim      = vecdim;
        args[t].lut_size    = lut_size;
        args[t].q_start     = t * chunk;
        args[t].q_end       = std::min(num_queries, (t + 1) * chunk);
        args[t].all_dtables = all_dtables;
        pthread_create(&tids[t], nullptr, _pq_lut_worker, &args[t]);
    }
    for (int t = 0; t < num_threads; ++t) pthread_join(tids[t], nullptr);
}


// =============================================================================
// OpenMP version
// =============================================================================

// pq_batch_build_lut_omp — OMP counterpart of pq_batch_build_lut_pthread
//
// #pragma omp parallel for schedule(static): all queries do identical work
// (M × K/16 outer iters × dsub inner iters), so static chunks balance perfectly.
// OMP thread pool persists across calls — zero per-call spawn overhead.
void pq_batch_build_lut_omp(
    const PQIndexSIMD& pq_simd,
    const float* queries, int num_queries, size_t vecdim,
    float* all_dtables,
    int num_threads)
{
    const size_t lut_size = pq_simd.idx->M * pq_simd.idx->K;
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int i = 0; i < num_queries; ++i) 
    {
        pq_build_lut_cc_unroll(
            pq_simd,
            queries     + (size_t)i * vecdim,
            all_dtables + (size_t)i * lut_size);
    }
}
