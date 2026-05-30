// =============================================================================
// ivf_hnsw.h -- IVF+HNSW two-level index (no MPI, no OMP)
//
// Architecture:
//   Coarse phase: IVF centroid scan selects top-nprobe clusters (same as
//                 ivf_search_simd Phase 1).
//   Fine phase:   Each selected cluster has its own small HierarchicalNSW.
//                 HNSW beam search replaces the flat SIMD scan.
//
// Build:
//   ivf_hnsw_build() -- runs IVF k-means, then builds one HNSW per cluster.
//   Labels stored in each per-cluster HNSW are the original base indices,
//   so ivf_hnsw_search() returns original IDs directly.
//
// Search:
//   ivf_hnsw_search() -- coarse IVF scan + HNSW fine search + k-merge.
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

#include "ivf_flat_simd.h"   // IVFIndex, ivf_build, ivf_load/save
#include "hnsw_simd.h"       // InnerProductSpaceNEON, HierarchicalNSW, hnsw_search_simd
#include "flat_simd.h"       // simd_inner_product_neon_unroll


// =============================================================================
// IVFHNSWIndex -- holds the IVF centroids + one HNSW per cluster
// =============================================================================
struct IVFHNSWIndex {
    IVFIndex                          ivf;
    InnerProductSpaceNEON*            space    = nullptr;
    std::vector<HierarchicalNSW<float>*> clusters; // clusters[c] is HNSW for cluster c
    size_t                            M              = 16;
    size_t                            ef_construction = 200;
};


// =============================================================================
// ivf_hnsw_build -- build IVF+HNSW index from scratch
//
// Parameters:
//   idx            : output IVFHNSWIndex (modified in-place)
//   base           : float base vectors [n x d], row-major
//   n              : number of base vectors
//   d              : vector dimension
//   nlist          : number of IVF clusters (k-means centroids)
//   n_iters        : Lloyd iterations for k-means
//   M              : HNSW links per node
//   ef_construction: HNSW build beam width
// =============================================================================
void ivf_hnsw_build(IVFHNSWIndex& idx,
                    const float*  base,
                    size_t        n,
                    size_t        d,
                    size_t        nlist        = 1024,
                    int           n_iters      = 25,
                    size_t        M            = 16,
                    size_t        ef_construction = 200)
{
    idx.M               = M;
    idx.ef_construction = ef_construction;

    // --- Step 1: build IVF (flat k-means, no reorder needed here) ------------
    ivf_build(idx.ivf, base, n, d, nlist, n_iters, /*reorder=*/false);

    // --- Step 2: allocate shared NEON space interface ------------------------
    idx.space = new InnerProductSpaceNEON(d);

    // --- Step 3: build one HNSW per cluster ----------------------------------
    idx.clusters.assign(nlist, nullptr);

    for (size_t c = 0; c < nlist; ++c) {
        const std::vector<uint32_t>& inv = idx.ivf.invlists[c];
        if (inv.empty()) continue;

        size_t csz  = inv.size();
        auto*  hnsw = new HierarchicalNSW<float>(
            idx.space, csz, M, ef_construction);

        // Insert each cluster member; label = original base index
        for (size_t j = 0; j < csz; ++j)
            hnsw->addPoint(base + (size_t)inv[j] * d,
                           static_cast<hnswlib::labeltype>(inv[j]));

        idx.clusters[c] = hnsw;
    }
}


// =============================================================================
// ivf_hnsw_search -- query the IVF+HNSW index for a single query vector
//
// Phase 1 (coarse): rank all nlist centroids by inner-product distance,
//                   select top-nprobe clusters.
// Phase 2 (fine):   beam-search the per-cluster HNSW with width ef,
//                   merge results into global top-k heap.
//
// Parameters:
//   idx    : built IVFHNSWIndex
//   query  : float query vector [vecdim]
//   k      : number of neighbors to return
//   nprobe : clusters to visit in phase 2 (latency-recall knob)
//   ef     : HNSW beam width in phase 2 (latency-recall knob)
//
// Returns: max-heap of (distance, original_base_index), size <= k.
// =============================================================================
std::priority_queue<std::pair<float, uint32_t>>
ivf_hnsw_search(const IVFHNSWIndex& idx,
                const float*        query,
                size_t              k,
                size_t              nprobe,
                size_t              ef)
{
    const size_t d     = idx.ivf.vecdim;
    const size_t nlist = idx.ivf.nlist;
    const size_t np    = std::min(nprobe, nlist);

    // --- Phase 1: coarse centroid scan ---------------------------------------
    std::vector<std::pair<float, uint32_t>> coarse(nlist);
    for (size_t c = 0; c < nlist; ++c) {
        float ip = simd_inner_product_neon_unroll(
            idx.ivf.centroids.data() + c * d, query, d);
        coarse[c] = {1.0f - ip, static_cast<uint32_t>(c)};
    }
    std::partial_sort(coarse.begin(), coarse.begin() + np, coarse.end());

    // --- Phase 2: HNSW fine search per selected cluster ---------------------
    // Global max-heap keeps the current top-k across all probed clusters.
    std::priority_queue<std::pair<float, uint32_t>> global_heap;

    for (size_t pi = 0; pi < np; ++pi) {
        uint32_t              c    = coarse[pi].second;
        HierarchicalNSW<float>* hnsw = idx.clusters[c];
        if (!hnsw) continue;

        // ef_used must be >= k; hnsw_search_simd enforces this internally
        auto local_res = hnsw_search_simd(hnsw, query, k, ef);

        // Merge local_res into global_heap, keeping only top-k
        while (!local_res.empty()) {
            auto [dist, orig] = local_res.top();
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
// ivf_hnsw_free -- release all heap memory owned by the index
// =============================================================================
void ivf_hnsw_free(IVFHNSWIndex& idx)
{
    for (auto* h : idx.clusters) delete h;
    idx.clusters.clear();
    delete idx.space;
    idx.space = nullptr;
}


// =============================================================================
// ivf_hnsw_save -- persist the full IVF+HNSW index to a directory
//
// Layout inside `dir`:
//   ivf.bin          -- IVF centroids + invlists (via ivf_save)
//   c0000.hnsw ...   -- one hnswlib binary file per non-empty cluster
//
// Returns true on success.
// =============================================================================
static bool ivf_hnsw_save(const IVFHNSWIndex& idx, const std::string& dir)
{
    mkdir(dir.c_str(), 0755);

    // IVF part
    std::string ivf_path = dir + "/ivf.bin";
    if (!ivf_save(idx.ivf, ivf_path.c_str())) return false;

    // Per-cluster HNSW files
    for (size_t c = 0; c < idx.clusters.size(); ++c) {
        if (!idx.clusters[c]) continue;
        char name[32];
        snprintf(name, sizeof(name), "/c%04zu.hnsw", c);
        idx.clusters[c]->saveIndex(dir + name);
    }
    return true;
}


// =============================================================================
// ivf_hnsw_load -- restore an IVF+HNSW index previously saved with ivf_hnsw_save
//
// Returns true if every expected file was found and loaded, false otherwise.
// =============================================================================
static bool ivf_hnsw_load(IVFHNSWIndex& idx, const std::string& dir,
                           size_t M, size_t ef_construction)
{
    // IVF part
    std::string ivf_path = dir + "/ivf.bin";
    if (!ivf_load(idx.ivf, ivf_path.c_str())) return false;

    idx.M               = M;
    idx.ef_construction = ef_construction;
    idx.space           = new InnerProductSpaceNEON(idx.ivf.vecdim);

    size_t nlist = idx.ivf.nlist;
    idx.clusters.assign(nlist, nullptr);

    for (size_t c = 0; c < nlist; ++c) {
        if (idx.ivf.invlists[c].empty()) continue;
        char name[32];
        snprintf(name, sizeof(name), "/c%04zu.hnsw", c);
        std::string path = dir + name;

        // Check file exists before constructing
        { std::ifstream chk(path, std::ios::binary); if (!chk.good()) return false; }

        idx.clusters[c] = new HierarchicalNSW<float>(idx.space, path);
    }
    return true;
}
