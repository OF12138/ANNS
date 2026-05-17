// =============================================================================
// flat_simd_omp.h — OpenMP-parallelized flat k-NN search (AArch64 NEON)
//
// OpenMP counterparts of the two strategies in flat_simd_pthread.h.
// Both reuse the simd_inner_product_neon_unroll / simd_flat_search_unroll
// kernels from flat_simd.h — only the threading model changes.
//
// ── Strategy 1: Query-level parallelism (OpenMP) ──────────────────────────────
// simd_flat_search_query_parallel_omp(base, queries, ..., results, num_threads)
//
//   #pragma omp parallel for over the query index. Each iteration calls
//   simd_flat_search_unroll on a distinct query and writes to a distinct slot
//   of results[]. schedule(static) — work per query is uniform.
//   No critical section needed (disjoint writes).
//   Metric affected: THROUGHPUT (queries/sec).
//
// ── Strategy 2: Base-partition parallelism (OpenMP) ───────────────────────────
// simd_flat_search_base_parallel_omp(base, query, ..., num_threads)
//
//   For a SINGLE query: #pragma omp parallel splits base[0..base_number) by
//   omp_get_thread_num()/omp_get_num_threads(). Each thread builds a local
//   max-heap of size k. After the parallel region, the master merges all
//   local heaps into the global top-k. recall@k = 1.0 (exact).
//   Metric affected: per-query LATENCY.
//
// OpenMP vs Pthread:
//   Pthread variant creates/joins threads on every call.
//   OpenMP keeps a long-lived thread pool — lower per-call overhead, especially
//   visible on small workloads or when the function is invoked in a hot loop.
//
// Platform: AArch64. Compile: g++ main.cc -o main -O2 -fopenmp -lpthread -std=c++11
// =============================================================================
#pragma once
#include <omp.h>
#include <queue>
#include <utility>
#include <vector>
#include <cstdint>
#include "flat_simd.h"   // simd_flat_search_unroll, simd_inner_product_neon_unroll


// =============================================================================
// Strategy 1 — Query-level parallelism (OpenMP)
// =============================================================================

// simd_flat_search_query_parallel_omp — batch k-NN, queries split via OpenMP for
//
// Parameters mirror simd_flat_search_query_parallel (pthread version).
//   results : caller-allocated array of num_queries priority_queues (output)
//   num_threads : pinned thread count for this parallel region
//
// Algorithm:
//   #pragma omp parallel for: each iteration i runs simd_flat_search_unroll
//   on queries[i*vecdim ..] and writes its top-k heap to results[i].
//   schedule(static) — every query does identical work, so even chunks balance
//   perfectly and avoid scheduler overhead.
void simd_flat_search_query_parallel_omp(
    float* base, float* queries,
    size_t base_number, size_t vecdim, size_t k,
    int num_queries,
    std::priority_queue<std::pair<float, uint32_t>>* results,
    int num_threads)
{
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int i = 0; i < num_queries; ++i) 
    {
        results[i] = simd_flat_search_unroll(
            base, queries + (size_t)i * vecdim,
            base_number, vecdim, k);
    }
}


// =============================================================================
// Strategy 2 — Base-partition parallelism (OpenMP)
// =============================================================================

// simd_flat_search_base_parallel_omp — single-query k-NN, base split via OpenMP
//
// Algorithm:
//   1. #pragma omp parallel: each thread t computes its segment
//      [t*chunk, min((t+1)*chunk, base_number)) using NEON dot product,
//      maintains a local max-heap of size k, and stores it in local_heaps[t].
//   2. After the parallel region (implicit barrier), master merges all local
//      heaps into the global top-k.
//
// Correctness:
//   True global top-k must reside in at least one segment's local top-k,
//   so the merged result is exact (recall@k = 1.0).
//
// Returns: max-heap of (IP distance, vector index) pairs — same shape as
//          simd_flat_search_unroll.
std::priority_queue<std::pair<float, uint32_t>>
simd_flat_search_base_parallel_omp(
    float* base, float* query,
    size_t base_number, size_t vecdim, size_t k,
    int num_threads)
{
    using _HeapT = std::priority_queue<std::pair<float, uint32_t>>;
    std::vector<_HeapT> local_heaps(num_threads);

    #pragma omp parallel num_threads(num_threads)
    {
        int tid    = omp_get_thread_num();
        int nth    = omp_get_num_threads();
        size_t chunk      = (base_number + (size_t)nth - 1) / (size_t)nth;
        size_t base_start = (size_t)tid * chunk;
        size_t base_end   = base_start + chunk;
        if (base_end > base_number) base_end = base_number;

        _HeapT& h = local_heaps[tid];
        for (size_t i = base_start; i < base_end; ++i) 
        {
            float dis = 1.0f - simd_inner_product_neon_unroll(base + i * vecdim, query, vecdim);
            if (h.size() < k)  h.push({dis, (uint32_t)i});
            else if (dis < h.top().first) 
            {
                h.push({dis, (uint32_t)i});
                h.pop();
            }
        }
    }

    // ── Merge phase ───────────────────────────────────────────────────────────
    _HeapT result;
    for (int t = 0; t < num_threads; ++t) {
        while (!local_heaps[t].empty()) {
            auto top = local_heaps[t].top();
            local_heaps[t].pop();
            if (result.size() < k) {
                result.push(top);
            } else if (top.first < result.top().first) {
                result.push(top);
                result.pop();
            }
        }
    }
    return result;
}
