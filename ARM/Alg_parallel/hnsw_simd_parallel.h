// =============================================================================
// hnsw_simd_parallel.h — Multi-entry-point parallel HNSW search
//
// Strategy: multiple independent entry points (多入口点并行)
//   T threads each run a complete layer-0 beam search from a DIFFERENT starting
//   node.  Each search is fully independent — no shared writes, no locks during
//   the search phase.  Results from all T threads are merged in the main thread
//   to produce the global top-k.
//
// Why this works:
//   HNSW beam search is a greedy local exploration.  Starting from diverse entry
//   points causes each thread to traverse a different region of the graph, so
//   the union of T searches covers substantially more nodes than one search at
//   the same efSearch.  Recall improves without increasing wall-clock latency
//   (T threads run in parallel on T cores, each taking ≈ single-thread time).
//
// Entry point selection:
//   Thread 0  : enterpoint_node_  (standard global entry — graph's "highest" node)
//   Thread t  : t * (base_number / T)  (uniform spread across node ID space)
//   The uniform spread gives structural diversity: nodes far apart in ID space
//   were inserted at different graph-build phases and have different neighborhoods.
//
// Thread safety:
//   searchBaseLayerST<true> (bare-bone path) is read-only on data_level0_memory_
//   and linkLists_.  VisitedListPool::getFreeVisitedList() is mutex-guarded and
//   allocates a new VisitedList when the pool is exhausted, so T concurrent
//   calls each get a private marker array.  No shared mutable state remains.
//
// Variants:
//   hnsw_search_multi_entry_pthread (SEARCH_ALG 31) — one Pthread per entry point
//   hnsw_search_multi_entry_omp     (SEARCH_ALG 32) — OpenMP parallel for
//
// Platform: AArch64.
// Compile:  g++ main.cc -o main -O2 -fopenmp -lpthread -std=c++11
// =============================================================================
#pragma once
#include <pthread.h>
#include <omp.h>
#include <queue>
#include <set>
#include <vector>
#include <utility>
#include <cstdint>
#include <algorithm>
#include "hnsw_simd.h"   // InnerProductSpaceNEON, hnsw_search_simd, HierarchicalNSW

using namespace hnswlib;


// =============================================================================
// _hnsw_search_from_ep — layer-0 beam search from an explicit entry point
//
// Identical to hnsw_search_simd but uses ep_id as the beam-search start node
// instead of appr_alg->enterpoint_node_.  Thread-safe: each call gets its own
// VisitedList from the pool.
//
// Parameters:
//   appr_alg : index (read-only during search)
//   query    : float query vector
//   k        : results to return
//   ef       : beam width; clamped to max(ef, k)
//   ep_id    : internal node ID to start beam search from
// =============================================================================
static std::priority_queue<std::pair<float, uint32_t>>
_hnsw_search_from_ep(HierarchicalNSW<float>* appr_alg,
                     const float* query, size_t k, size_t ef,
                     tableint ep_id)
{
    const size_t ef_used = std::max(ef, k);

    auto top_candidates = appr_alg->searchBaseLayerST<true>(
        ep_id, static_cast<const void*>(query), ef_used);

    while (top_candidates.size() > k)
        top_candidates.pop();

    std::priority_queue<std::pair<float, uint32_t>> result;
    while (!top_candidates.empty()) {
        auto p = top_candidates.top();
        top_candidates.pop();
        result.push({p.first,
                     static_cast<uint32_t>(appr_alg->getExternalLabel(p.second))});
    }
    return result;
}


// =============================================================================
// _hnsw_merge_results — merge T per-thread top-k heaps into one global top-k
//
// Each results[t] is a max-heap of (distance, id), size ≤ k.
// The merge keeps the k smallest distances (closest neighbors) overall.
// Distance convention: 1.0 - IP, so smaller = closer = better.
// Max-heap keeps worst (largest dist) at top; pop when size > k.
// =============================================================================
static std::priority_queue<std::pair<float, uint32_t>>
_hnsw_merge_results(
    std::vector<std::priority_queue<std::pair<float, uint32_t>>>& results,
    int T, size_t k)
{
    std::priority_queue<std::pair<float, uint32_t>> merged;
    for (int t = 0; t < T; ++t) {
        while (!results[t].empty()) {
            auto item = results[t].top();
            results[t].pop();
            if (merged.size() < k) {
                merged.push(item);
            } else if (item.first < merged.top().first) {
                merged.pop();
                merged.push(item);
            }
        }
    }
    return merged;
}


// =============================================================================
// _hnsw_merge_results_dedup — same as _hnsw_merge_results but skips duplicate
// external IDs.  Required when T threads explore the same small graph and
// converge to overlapping candidate sets: without dedup, a duplicate ID can
// displace a unique good result in the merged heap, shrinking effective k.
// =============================================================================
static std::priority_queue<std::pair<float, uint32_t>>
_hnsw_merge_results_dedup(
    std::vector<std::priority_queue<std::pair<float, uint32_t>>>& results,
    int T, size_t k)
{
    std::priority_queue<std::pair<float, uint32_t>> merged;
    std::set<uint32_t> seen;
    for (int t = 0; t < T; ++t) {
        while (!results[t].empty()) {
            auto item = results[t].top();
            results[t].pop();
            if (!seen.insert(item.second).second) continue;
            if (merged.size() < k) {
                merged.push(item);
            } else if (item.first < merged.top().first) {
                merged.pop();
                merged.push(item);
            }
        }
    }
    return merged;
}


// =============================================================================
// hnsw_search_multi_entry_omp_v2 — multi-entry OMP search with dedup merge
//
// Identical to hnsw_search_multi_entry_omp except the T result heaps are
// merged with _hnsw_merge_results_dedup, guaranteeing k unique IDs in the
// output even when threads' candidate sets overlap (common on small graphs).
// =============================================================================
std::priority_queue<std::pair<float, uint32_t>>
hnsw_search_multi_entry_omp_v2(
    HierarchicalNSW<float>* appr_alg,
    const float* query, size_t k, size_t ef,
    size_t base_number, int num_threads)
{
    const int T = num_threads;
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> results(T);

    #pragma omp parallel for num_threads(T) schedule(static, 1)
    for (int t = 0; t < T; ++t) {
        tableint ep = (t == 0)
            ? appr_alg->enterpoint_node_
            : static_cast<tableint>((size_t)t * base_number / T);
        results[t] = _hnsw_search_from_ep(appr_alg, query, k, ef, ep);
    }

    return _hnsw_merge_results_dedup(results, T, k);
}


// =============================================================================
// _HNSWMultiEntryArgs — per-thread argument block for Pthread version
// =============================================================================
struct _HNSWMultiEntryArgs {
    HierarchicalNSW<float>* appr_alg;
    const float*  query;
    size_t        k;
    size_t        ef;
    tableint      ep_id;
    std::priority_queue<std::pair<float, uint32_t>> result;
};

static void* _hnsw_multi_entry_worker(void* arg)
{
    auto* a = static_cast<_HNSWMultiEntryArgs*>(arg);
    a->result = _hnsw_search_from_ep(a->appr_alg, a->query, a->k, a->ef, a->ep_id);
    return nullptr;
}


// =============================================================================
// hnsw_search_multi_entry_pthread — multi-entry-point search, Pthread version
//
// Spawns num_threads Pthreads; each runs one complete beam search from a
// different entry point.  After all threads join, merges T result heaps.
//
// Parameters:
//   appr_alg    : built HierarchicalNSW<float> index (read-only)
//   query       : float query vector
//   k           : final top-k results
//   ef          : beam width per thread (efSearch)
//   base_number : total number of nodes in the graph (for entry-point spread)
//   num_threads : number of Pthread workers (= number of independent searches)
// =============================================================================
std::priority_queue<std::pair<float, uint32_t>>
hnsw_search_multi_entry_pthread(
    HierarchicalNSW<float>* appr_alg,
    const float* query, size_t k, size_t ef,
    size_t base_number, int num_threads)
{
    const int T = num_threads;

    std::vector<pthread_t>            tids(T);
    std::vector<_HNSWMultiEntryArgs>  args(T);

    for (int t = 0; t < T; ++t) {
        // Thread 0 uses the standard global entry point for a fair baseline.
        // Threads 1..T-1 spread uniformly across node ID space.
        tableint ep = (t == 0)
            ? appr_alg->enterpoint_node_
            : static_cast<tableint>((size_t)t * base_number / T);

        args[t].appr_alg = appr_alg;
        args[t].query    = query;
        args[t].k        = k;
        args[t].ef       = ef;
        args[t].ep_id    = ep;
        pthread_create(&tids[t], nullptr, _hnsw_multi_entry_worker, &args[t]);
    }

    for (int t = 0; t < T; ++t)
        pthread_join(tids[t], nullptr);

    // Collect per-thread result heaps for merging
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> results(T);
    for (int t = 0; t < T; ++t)
        results[t] = std::move(args[t].result);

    return _hnsw_merge_results(results, T, k);
}


// =============================================================================
// hnsw_search_multi_entry_omp — multi-entry-point search, OpenMP version
//
// Uses OpenMP parallel for with schedule(static, 1): each of the T loop
// iterations runs on a different thread, one entry point per iteration.
// OpenMP's persistent thread pool avoids per-query pthread_create overhead,
// giving lower overhead than the Pthread version especially at high T.
//
// Parameters: same as hnsw_search_multi_entry_pthread.
// =============================================================================
std::priority_queue<std::pair<float, uint32_t>>
hnsw_search_multi_entry_omp(
    HierarchicalNSW<float>* appr_alg,
    const float* query, size_t k, size_t ef,
    size_t base_number, int num_threads)
{
    const int T = num_threads;
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> results(T);

    #pragma omp parallel for num_threads(T) schedule(static, 1)
    for (int t = 0; t < T; ++t) {
        tableint ep = (t == 0)
            ? appr_alg->enterpoint_node_
            : static_cast<tableint>((size_t)t * base_number / T);
        results[t] = _hnsw_search_from_ep(appr_alg, query, k, ef, ep);
    }

    return _hnsw_merge_results(results, T, k);
}
