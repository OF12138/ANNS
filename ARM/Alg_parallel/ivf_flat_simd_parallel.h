// =============================================================================
// ivf_flat_simd_parallel.h — Cluster-partition parallel IVF fine scan (Pthread)
//
// Parallelizes the fine-scan phase of IVF across a pool of Pthreads.
// The coarse phase (nlist centroid rankings) runs single-thread — at nlist=1024
// it costs ~10 µs; thread spawn overhead (~350 µs) would far exceed any gain.
//
// Fine-scan parallelism — two work-distribution strategies (IVF_FLATTEN):
//
//   IVF_FLATTEN 0 — cluster-split:
//     Assign the nprobe selected clusters to threads in round-robin order.
//     Thread t scans clusters at positions t, t+nth, t+2*nth, ... in the probe
//     list. Simple, zero extra allocation. Non-uniform load when cluster sizes
//     differ significantly.
//
//   IVF_FLATTEN 1 — flatten-then-split (default):
//     Concatenate all original vector IDs from every selected cluster into a
//     flat array, then split the array evenly across threads by vector count.
//     Perfectly balanced by construction. Extra O(nprobe × avg_cluster_size)
//     temporary allocation (~1600 uint32_t for nprobe=16, nlist=1024 on DEEP100K).
//
// Thread lifetime per query call: create num_threads, join num_threads, merge.
// Pthreads are NOT persistent across query calls — overhead is ~350–700 µs per
// call. At small nprobe the fine scan is ≪ 1 ms, so negative speedup is expected.
// Positive speedup requires nprobe large enough that fine-scan time >> spawn cost.
//
// Platform: AArch64. Compile: g++ main.cc -o main -O2 -fopenmp -lpthread -std=c++11
// =============================================================================
#pragma once
#include <sys/time.h>
#include <pthread.h>
#include <queue>
#include <utility>
#include <vector>
#include <algorithm>
#include <cstdint>
#include "ivf_flat_simd.h"    // IVFIndex
#include "flat_simd.h"        // simd_inner_product_neon_unroll

// IVF_FLATTEN — work distribution strategy for the parallel fine scan.
//   0 = cluster-split    (non-uniform, no extra alloc)
//   1 = flatten-then-split (uniform load, small temporary alloc) [default]
#ifndef IVF_FLATTEN
#define IVF_FLATTEN 1
#endif


// =============================================================================
// Worker argument struct and thread function
// =============================================================================

struct _IVFWorkerArgs {
    const IVFIndex* idx;
    const float*    base;
    const float*    query;
    size_t          d;
    size_t          k;

    // IVF_FLATTEN == 0: cluster-split fields
    const uint32_t* probe_ids;   // selected cluster IDs [nprobe]
    size_t          nprobe;
    int             tid;
    int             nth;

    // IVF_FLATTEN == 1: flatten fields
    const uint32_t* flat_ids;    // original vector IDs from all selected clusters
    const size_t*   flat_pos;    // reordered_base slot positions (null if not reordered)
    size_t          f_start;     // inclusive
    size_t          f_end;       // exclusive

    // per-thread output
    std::priority_queue<std::pair<float, uint32_t>> local_heap;
};

static void* _ivf_cluster_worker(void* arg)
{
    auto* a = static_cast<_IVFWorkerArgs*>(arg);
    auto& heap       = a->local_heap;
    const float* q   = a->query;
    const size_t d   = a->d;
    const size_t k   = a->k;
    const IVFIndex& idx = *a->idx;

#if IVF_FLATTEN
    // ── Flatten mode: scan a contiguous slice of the flat vector ID array ─────
    if (a->flat_pos != nullptr) {
        // reordered: use sequential reordered_base access (cache-friendly)
        for (size_t j = a->f_start; j < a->f_end; ++j) {
            size_t   pos  = a->flat_pos[j];
            uint32_t orig = a->flat_ids[j];
            float ip  = simd_inner_product_neon_unroll(
                idx.reordered_base.data() + pos * d, q, d);
            float dis = 1.0f - ip;
            if (heap.size() < k) {
                heap.push({dis, orig});
            } else if (dis < heap.top().first) {
                heap.push({dis, orig});
                heap.pop();
            }
        }
    } else {
        // non-reordered: random access into original base
        for (size_t j = a->f_start; j < a->f_end; ++j) {
            uint32_t orig = a->flat_ids[j];
            float ip  = simd_inner_product_neon_unroll(
                a->base + (size_t)orig * d, q, d);
            float dis = 1.0f - ip;
            if (heap.size() < k) {
                heap.push({dis, orig});
            } else if (dis < heap.top().first) {
                heap.push({dis, orig});
                heap.pop();
            }
        }
    }
#else
    // ── Cluster-split mode: round-robin cluster assignment ────────────────────
    // Thread tid handles clusters at positions: tid, tid+nth, tid+2*nth, ...
    for (size_t ci = (size_t)a->tid; ci < a->nprobe; ci += (size_t)a->nth) {
        uint32_t c = a->probe_ids[ci];
        if (idx.reordered) {
            size_t start = idx.cluster_offset[c];
            size_t end   = idx.cluster_offset[c + 1];
            for (size_t j = start; j < end; ++j) {
                float ip  = simd_inner_product_neon_unroll(
                    idx.reordered_base.data() + j * d, q, d);
                float dis = 1.0f - ip;
                uint32_t orig = idx.invlists[c][j - start];
                if (heap.size() < k) {
                    heap.push({dis, orig});
                } else if (dis < heap.top().first) {
                    heap.push({dis, orig});
                    heap.pop();
                }
            }
        } else {
            for (uint32_t orig : idx.invlists[c]) {
                float ip  = simd_inner_product_neon_unroll(
                    a->base + (size_t)orig * d, q, d);
                float dis = 1.0f - ip;
                if (heap.size() < k) {
                    heap.push({dis, orig});
                } else if (dis < heap.top().first) {
                    heap.push({dis, orig});
                    heap.pop();
                }
            }
        }
    }
#endif
    return nullptr;
}


// =============================================================================
// ivf_search_simd_cluster_pthread
//
// Two-phase IVF query with cluster-partition parallelism in the fine scan.
//
// Parameters:
//   idx         : built IVFIndex
//   base        : original base dataset (needed for IVF_FLATTEN=1 and non-reordered)
//   query       : single query vector [vecdim]
//   k           : top-k results
//   nprobe      : number of clusters to fine-scan (latency-recall knob)
//   num_threads : Pthread worker count for the fine scan
//
// Phase 1 (single-thread): NEON IP against all nlist centroids → partial_sort
//   top-nprobe.  Not parallelized: at nlist=1024 this takes ~10 µs; Pthread
//   spawn cost (~350 µs) would dominate.
// Phase 2 (parallel): num_threads threads scan the selected clusters.
//   IVF_FLATTEN=0 → round-robin cluster assignment (may be unbalanced)
//   IVF_FLATTEN=1 → flat vector list, split evenly by vector count
// After join: merge all local top-k heaps into the global top-k.
// =============================================================================
std::priority_queue<std::pair<float, uint32_t>>
ivf_search_simd_cluster_pthread(
    const IVFIndex& idx, const float* base,
    const float* query, size_t k, size_t nprobe, int num_threads)
{
    const size_t d  = idx.vecdim;
    const size_t np = std::min(nprobe, idx.nlist);

    // ── Phase 1: Coarse (single-thread) ──────────────────────────────────────
    std::vector<std::pair<float, uint32_t>> coarse(idx.nlist);
    for (size_t c = 0; c < idx.nlist; ++c) {
        float ip = simd_inner_product_neon_unroll(
            idx.centroids.data() + c * d, query, d);
        coarse[c] = {1.0f - ip, static_cast<uint32_t>(c)};
    }
    std::partial_sort(coarse.begin(), coarse.begin() + np, coarse.end());

    std::vector<uint32_t> probe_ids(np);
    for (size_t i = 0; i < np; ++i) probe_ids[i] = coarse[i].second;

    // ── Phase 2: Fine (parallel) ──────────────────────────────────────────────
    std::vector<pthread_t>      tids(num_threads);
    std::vector<_IVFWorkerArgs> args(num_threads);

#if IVF_FLATTEN
    // Build flat list: concatenate invlists of all selected clusters
    size_t total = 0;
    for (size_t i = 0; i < np; ++i)
        total += idx.invlists[probe_ids[i]].size();

    std::vector<uint32_t> flat_ids;
    std::vector<size_t>   flat_pos;   // reordered slot positions; empty when not reordered
    flat_ids.reserve(total);
    if (idx.reordered) flat_pos.reserve(total);

    for (size_t i = 0; i < np; ++i) {
        uint32_t c = probe_ids[i];
        const size_t base_pos = idx.reordered ? idx.cluster_offset[c] : 0;
        for (size_t j = 0; j < idx.invlists[c].size(); ++j) {
            flat_ids.push_back(idx.invlists[c][j]);
            if (idx.reordered) flat_pos.push_back(base_pos + j);
        }
    }

    size_t chunk = (total + (size_t)num_threads - 1) / (size_t)num_threads;
    for (int t = 0; t < num_threads; ++t) {
        args[t].idx      = &idx;
        args[t].base     = base;
        args[t].query    = query;
        args[t].d        = d;
        args[t].k        = k;
        args[t].flat_ids = flat_ids.data();
        args[t].flat_pos = idx.reordered ? flat_pos.data() : nullptr;
        args[t].f_start  = std::min((size_t)t * chunk, total);
        args[t].f_end    = std::min((size_t)(t + 1) * chunk, total);
        pthread_create(&tids[t], nullptr, _ivf_cluster_worker, &args[t]);
    }
#else
    for (int t = 0; t < num_threads; ++t) {
        args[t].idx       = &idx;
        args[t].base      = base;
        args[t].query     = query;
        args[t].d         = d;
        args[t].k         = k;
        args[t].probe_ids = probe_ids.data();
        args[t].nprobe    = np;
        args[t].tid       = t;
        args[t].nth       = num_threads;
        pthread_create(&tids[t], nullptr, _ivf_cluster_worker, &args[t]);
    }
#endif

    for (int t = 0; t < num_threads; ++t) pthread_join(tids[t], nullptr);

    // ── Merge local heaps → global top-k ─────────────────────────────────────
    std::priority_queue<std::pair<float, uint32_t>> result;
    for (int t = 0; t < num_threads; ++t) {
        auto& lh = args[t].local_heap;
        while (!lh.empty()) {
            auto top = lh.top(); lh.pop();
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


// =============================================================================
// ivf_search_simd_cluster_pthread_timed
//
// Identical to ivf_search_simd_cluster_pthread but accumulates per-phase
// wall-clock times into three caller-owned int64_t counters (in µs):
//
//   t_coarse_us : centroid IP computation + partial_sort (Phase 1, single-thread)
//   t_scan_us   : flat-list build (if IVF_FLATTEN=1) + thread create/work/join
//   t_merge_us  : merging num_threads local top-k heaps into the global top-k
//
// Accumulators are *added to* on each call so the caller can sum over all
// queries and divide by test_number to obtain per-query averages.
// =============================================================================
std::priority_queue<std::pair<float, uint32_t>>
ivf_search_simd_cluster_pthread_timed(
    const IVFIndex& idx, const float* base,
    const float* query, size_t k, size_t nprobe, int num_threads,
    int64_t* t_coarse_us, int64_t* t_scan_us, int64_t* t_merge_us,
    int64_t* t_create_us, int64_t* t_join_us)
{
    struct timeval tp0, tp1;
    const size_t d  = idx.vecdim;
    const size_t np = std::min(nprobe, idx.nlist);

    // ── Phase 1: Coarse ───────────────────────────────────────────────────────
    gettimeofday(&tp0, NULL);
    std::vector<std::pair<float, uint32_t>> coarse(idx.nlist);
    for (size_t c = 0; c < idx.nlist; ++c) {
        float ip = simd_inner_product_neon_unroll(
            idx.centroids.data() + c * d, query, d);
        coarse[c] = {1.0f - ip, static_cast<uint32_t>(c)};
    }
    std::partial_sort(coarse.begin(), coarse.begin() + np, coarse.end());
    gettimeofday(&tp1, NULL);
    *t_coarse_us += (tp1.tv_sec * 1000000LL + tp1.tv_usec)
                  - (tp0.tv_sec * 1000000LL + tp0.tv_usec);

    std::vector<uint32_t> probe_ids(np);
    for (size_t i = 0; i < np; ++i) probe_ids[i] = coarse[i].second;

    // ── Phase 2: Scan (flat-list build + thread create/work/join) ─────────────
    gettimeofday(&tp0, NULL);
    std::vector<pthread_t>      tids(num_threads);
    std::vector<_IVFWorkerArgs> args(num_threads);

#if IVF_FLATTEN
    size_t total = 0;
    for (size_t i = 0; i < np; ++i)
        total += idx.invlists[probe_ids[i]].size();

    std::vector<uint32_t> flat_ids;
    std::vector<size_t>   flat_pos;   // reordered slot positions; empty when not reordered
    flat_ids.reserve(total);
    if (idx.reordered) flat_pos.reserve(total);

    for (size_t i = 0; i < np; ++i) {
        uint32_t c = probe_ids[i];
        const size_t base_pos = idx.reordered ? idx.cluster_offset[c] : 0;
        for (size_t j = 0; j < idx.invlists[c].size(); ++j) {
            flat_ids.push_back(idx.invlists[c][j]);
            if (idx.reordered) flat_pos.push_back(base_pos + j);
        }
    }

    size_t chunk = (total + (size_t)num_threads - 1) / (size_t)num_threads;
    struct timeval tc0, tc1, tj0, tj1;
    gettimeofday(&tc0, NULL);
    for (int t = 0; t < num_threads; ++t) {
        args[t].idx      = &idx;
        args[t].base     = base;
        args[t].query    = query;
        args[t].d        = d;
        args[t].k        = k;
        args[t].flat_ids = flat_ids.data();
        args[t].flat_pos = idx.reordered ? flat_pos.data() : nullptr;
        args[t].f_start  = std::min((size_t)t * chunk, total);
        args[t].f_end    = std::min((size_t)(t + 1) * chunk, total);
        pthread_create(&tids[t], nullptr, _ivf_cluster_worker, &args[t]);
    }
    gettimeofday(&tc1, NULL);
    *t_create_us += (tc1.tv_sec * 1000000LL + tc1.tv_usec)
                  - (tc0.tv_sec * 1000000LL + tc0.tv_usec);
#else
    struct timeval tc0, tc1, tj0, tj1;
    gettimeofday(&tc0, NULL);
    for (int t = 0; t < num_threads; ++t) {
        args[t].idx       = &idx;
        args[t].base      = base;
        args[t].query     = query;
        args[t].d         = d;
        args[t].k         = k;
        args[t].probe_ids = probe_ids.data();
        args[t].nprobe    = np;
        args[t].tid       = t;
        args[t].nth       = num_threads;
        pthread_create(&tids[t], nullptr, _ivf_cluster_worker, &args[t]);
    }
    gettimeofday(&tc1, NULL);
    *t_create_us += (tc1.tv_sec * 1000000LL + tc1.tv_usec)
                  - (tc0.tv_sec * 1000000LL + tc0.tv_usec);
#endif

    gettimeofday(&tj0, NULL);
    for (int t = 0; t < num_threads; ++t) pthread_join(tids[t], nullptr);
    gettimeofday(&tj1, NULL);
    *t_join_us += (tj1.tv_sec * 1000000LL + tj1.tv_usec)
                - (tj0.tv_sec * 1000000LL + tj0.tv_usec);

    gettimeofday(&tp1, NULL);
    *t_scan_us += (tp1.tv_sec * 1000000LL + tp1.tv_usec)
                - (tp0.tv_sec * 1000000LL + tp0.tv_usec);

    // ── Phase 3: Merge local heaps → global top-k ────────────────────────────
    gettimeofday(&tp0, NULL);
    std::priority_queue<std::pair<float, uint32_t>> result;
    for (int t = 0; t < num_threads; ++t) {
        auto& lh = args[t].local_heap;
        while (!lh.empty()) {
            auto top = lh.top(); lh.pop();
            if (result.size() < k) {
                result.push(top);
            } else if (top.first < result.top().first) {
                result.push(top);
                result.pop();
            }
        }
    }
    gettimeofday(&tp1, NULL);
    *t_merge_us += (tp1.tv_sec * 1000000LL + tp1.tv_usec)
                 - (tp0.tv_sec * 1000000LL + tp0.tv_usec);

    return result;
}
