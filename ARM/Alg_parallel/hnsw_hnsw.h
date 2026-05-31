// =============================================================================
// hnsw_hnsw.h -- Two-level HNSW index (HNSW on HNSW)
//
// Architecture:
//   Coarse phase: a small HNSW built on the nlist IVF centroids replaces the
//                 flat centroid scan used in IVF+HNSW.  Navigating this graph
//                 finds the top-nprobe nearest centroids in O(log nlist) hops
//                 instead of scanning all nlist centroids.
//   Fine phase:   identical to IVF+HNSW -- one HierarchicalNSW per cluster,
//                 beam search within each selected cluster.
//
// Trade-off vs IVF+HNSW:
//   + Coarse phase is sub-linear in nlist (useful when nlist >> 1024).
//   - Coarse HNSW is approximate; may miss the true nearest centroid.
//     Recall is bounded by the quality of the coarse navigation.
//
// Build:  hnsw_hnsw_build()  -- IVF k-means + per-cluster HNSWs + coarse HNSW.
// Save:   hnsw_hnsw_save()   -- writes IVF+HNSW files then coarse.hnsw last.
// Load:   hnsw_hnsw_load()   -- checks coarse.hnsw first (completion marker).
// Free:   hnsw_hnsw_free()   -- releases all heap allocations.
// Search: hnsw_hnsw_search() -- single-thread baseline search.
//
// Platform: AArch64.
// =============================================================================
#pragma once

#include <vector>
#include <queue>
#include <utility>
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <fstream>
#include <string>
#include <sys/stat.h>

#include "ivf_hnsw.h"    // IVFHNSWIndex, ivf_hnsw_build/save/load/free
#include "hnsw_simd.h"   // HierarchicalNSW, hnsw_search_simd, InnerProductSpaceNEON


// =============================================================================
// HNSWHNSWIndex
// =============================================================================
struct HNSWHNSWIndex {
    IVFHNSWIndex              inner;               // IVF centroids + per-cluster HNSWs
    InnerProductSpaceNEON*    coarse_space = nullptr;
    HierarchicalNSW<float>*   coarse_hnsw  = nullptr; // HNSW on centroids
    size_t coarse_M               = 16;
    size_t coarse_ef_construction = 200;
};


// =============================================================================
// hnsw_hnsw_build -- build IVF (k-means) + per-cluster HNSWs + coarse HNSW
// =============================================================================
void hnsw_hnsw_build(
    HNSWHNSWIndex& idx,
    const float* base, size_t n, size_t d,
    size_t nlist               = 1024,
    int    n_iters             = 25,
    size_t M                   = 16,
    size_t ef_construction     = 200,
    size_t coarse_M            = 16,
    size_t coarse_ef_constr    = 200)
{
    idx.coarse_M               = coarse_M;
    idx.coarse_ef_construction = coarse_ef_constr;

    // Step 1: IVF k-means + per-cluster HNSWs (reuse existing builder)
    ivf_hnsw_build(idx.inner, base, n, d, nlist, n_iters, M, ef_construction);

    // Step 2: build coarse HNSW on the nlist centroids
    idx.coarse_space = new InnerProductSpaceNEON(d);
    idx.coarse_hnsw  = new HierarchicalNSW<float>(
        idx.coarse_space, nlist, coarse_M, coarse_ef_constr);

    for (size_t c = 0; c < nlist; ++c)
        idx.coarse_hnsw->addPoint(
            idx.inner.ivf.centroids.data() + c * d,
            static_cast<hnswlib::labeltype>(c));
}


// =============================================================================
// hnsw_hnsw_search -- single-thread two-level HNSW search
//
// Phase 1 (coarse): HNSW search on centroid graph → top-nprobe centroids.
// Phase 2 (fine):   beam-search per-cluster HNSW, merge into global top-k.
// =============================================================================
std::priority_queue<std::pair<float, uint32_t>>
hnsw_hnsw_search(
    const HNSWHNSWIndex& idx,
    const float* query,
    size_t k, size_t nprobe,
    size_t ef_coarse, size_t ef_fine)
{
    const size_t np = std::min(nprobe, idx.inner.ivf.nlist);

    // Coarse: HNSW graph navigation on centroids
    auto coarse_heap = hnsw_search_simd(idx.coarse_hnsw, query, np, ef_coarse);

    // Sort by distance ascending (closest centroid first)
    std::vector<std::pair<float, uint32_t>> coarse_sorted;
    coarse_sorted.reserve(coarse_heap.size());
    while (!coarse_heap.empty()) {
        coarse_sorted.push_back(coarse_heap.top());
        coarse_heap.pop();
    }
    std::sort(coarse_sorted.begin(), coarse_sorted.end());

    // Fine: HNSW beam search per selected cluster
    std::priority_queue<std::pair<float, uint32_t>> global_heap;
    for (const auto& ci : coarse_sorted) {
        uint32_t c    = ci.second;
        auto*    hnsw = idx.inner.clusters[c];
        if (!hnsw) continue;

        auto local_res = hnsw_search_simd(hnsw, query, k, ef_fine);
        while (!local_res.empty()) {
            float    dist = local_res.top().first;
            uint32_t orig = local_res.top().second;
            local_res.pop();
            if (global_heap.size() < k) {
                global_heap.push({dist, orig});
            } else if (dist < global_heap.top().first) {
                global_heap.pop();
                global_heap.push({dist, orig});
            }
        }
    }
    return global_heap;
}


// =============================================================================
// hnsw_hnsw_free
// =============================================================================
void hnsw_hnsw_free(HNSWHNSWIndex& idx)
{
    delete idx.coarse_hnsw;  idx.coarse_hnsw  = nullptr;
    delete idx.coarse_space; idx.coarse_space = nullptr;
    ivf_hnsw_free(idx.inner);
}


// =============================================================================
// hnsw_hnsw_save -- persist to directory
//
// Layout:
//   ivf.bin          -- IVF centroids + invlists
//   c0000.hnsw ...   -- per-cluster HNSW files
//   coarse.hnsw      -- coarse centroid HNSW  (written LAST = completion marker)
// =============================================================================
static bool hnsw_hnsw_save(const HNSWHNSWIndex& idx, const std::string& dir)
{
    mkdir(dir.c_str(), 0755);
    // Inner IVF+HNSW files first
    if (!ivf_hnsw_save(idx.inner, dir)) return false;
    // Coarse HNSW last (acts as completion marker)
    idx.coarse_hnsw->saveIndex(dir + "/coarse.hnsw");
    return true;
}


// =============================================================================
// hnsw_hnsw_load -- restore from directory
//
// Returns false if coarse.hnsw or any inner file is missing.
// coarse.hnsw is checked FIRST: its absence means the save was incomplete.
// =============================================================================
static bool hnsw_hnsw_load(
    HNSWHNSWIndex& idx, const std::string& dir,
    size_t M, size_t ef_construction,
    size_t coarse_M, size_t coarse_ef_constr)
{
    // Completion marker check
    std::string coarse_path = dir + "/coarse.hnsw";
    { std::ifstream chk(coarse_path, std::ios::binary); if (!chk.good()) return false; }

    // Load inner IVF+HNSW
    if (!ivf_hnsw_load(idx.inner, dir, M, ef_construction)) return false;

    // Load coarse HNSW
    idx.coarse_M               = coarse_M;
    idx.coarse_ef_construction = coarse_ef_constr;
    idx.coarse_space = new InnerProductSpaceNEON(idx.inner.ivf.vecdim);
    idx.coarse_hnsw  = new HierarchicalNSW<float>(idx.coarse_space, coarse_path);
    return true;
}
