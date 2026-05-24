// =============================================================================
// hnsw_simd.h — HNSW layer-0 beam search with ARM NEON inner-product distance
//
// Replaces hnswlib's default scalar InnerProductDistance (used on AArch64
// because USE_SSE/USE_AVX are not defined) with simd_inner_product_neon_unroll,
// the 4-accumulator NEON dot-product from flat_simd.h.
//
// Search strategy: layer-0 only
//   Upper-layer greedy descent is skipped.  The beam search starts directly
//   from enterpoint_node_ (the global graph entry point stored in the index)
//   and explores layer 0 with width efSearch.  For DEEP100K (100K vectors,
//   maxlevel ≈ 3–4) upper-layer navigation adds only ~O(log N) trivial hops,
//   so skipping it has minimal recall impact while simplifying the code.
//
// Key parameters (defined in main.cc):
//   HNSW_M              : bidirectional link count per node (default 16)
//                         Higher → better recall, more memory & build time
//   HNSW_EF_CONSTRUCTION: beam width during index build (default 200)
//                         Higher → better graph quality, slower build
//   HNSW_EF_SEARCH      : beam width during query (default 50)
//                         Primary latency-recall knob; sweep for trade-off curve
//
// Platform: AArch64.
// Compile:  g++ main.cc -o main -O2 -fopenmp -lpthread -std=c++11
// =============================================================================
#pragma once
#include <queue>
#include <vector>
#include <utility>
#include <cstdint>
#include <algorithm>
#include "../../hnswlib/hnswlib/hnswlib.h"
#include "../../hnswlib/hnswlib/hnswalg.h"
#include "flat_simd.h"   // simd_inner_product_neon_unroll

using namespace hnswlib;


// =============================================================================
// _hnsw_neon_ip_dist — NEON inner-product distance, DISTFUNC<float> compatible
//
// Returns 1.0 - IP(v1, v2) using the 4-accumulator NEON dot product.
// The qty_ptr parameter holds a pointer to the vector dimension (size_t),
// exactly as InnerProductSpace passes it.
// =============================================================================
static float _hnsw_neon_ip_dist(const void* v1, const void* v2, const void* qty_ptr)
{
    size_t d = *static_cast<const size_t*>(qty_ptr);
    return 1.0f - simd_inner_product_neon_unroll(
        static_cast<const float*>(v1),
        static_cast<const float*>(v2), d);
}


// =============================================================================
// InnerProductSpaceNEON — SpaceInterface<float> backed by ARM NEON SIMD
//
// Drop-in replacement for hnswlib::InnerProductSpace on AArch64.
// Passes _hnsw_neon_ip_dist as the distance function; the dim_ member is
// stored in the object and its address is returned as dist_func_param_.
// The object must remain alive for the lifetime of the index.
// =============================================================================
class InnerProductSpaceNEON : public SpaceInterface<float> {
    size_t dim_;
public:
    explicit InnerProductSpaceNEON(size_t dim) : dim_(dim) {}

    size_t get_data_size()           override { return dim_ * sizeof(float); }
    DISTFUNC<float> get_dist_func()  override { return _hnsw_neon_ip_dist; }
    void* get_dist_func_param()      override { return &dim_; }
};


// =============================================================================
// hnsw_search_simd — layer-0 beam search with NEON distance (SEARCH_ALG 30)
//
// Calls searchBaseLayerST<true> (bare-bone fast path: no deletion checks,
// no filter functor) directly from the index entry point.  This confines the
// search entirely to layer 0.
//
// Parameters:
//   appr_alg : built HierarchicalNSW<float> index (non-null, read-only during query)
//   query    : float query vector, length = index dimension
//   k        : number of nearest neighbors to return
//   ef       : beam search width (efSearch); ef is clamped to max(ef, k)
//
// Returns:
//   max-heap of (distance, vector_id) pairs, size k.
//   distance = 1.0 - IP(query, base_vector)  (same convention as all other algs)
// =============================================================================
std::priority_queue<std::pair<float, uint32_t>>
hnsw_search_simd(HierarchicalNSW<float>* appr_alg,
                 const float* query, size_t k, size_t ef)
{
    const size_t ef_used = std::max(ef, k);

    // Layer-0 beam search from global entry point (skips upper-layer descent)
    auto top_candidates = appr_alg->searchBaseLayerST<true>(
        appr_alg->enterpoint_node_,
        static_cast<const void*>(query),
        ef_used);

    // top_candidates is a max-heap (CompareByFirst): top() = farthest.
    // Trim to k by popping the farthest until only k remain.
    while (top_candidates.size() > k)
        top_candidates.pop();

    // Convert (float dist, tableint internal_id) → (float dist, uint32_t label)
    // getExternalLabel maps internal node ID → original base index [0, N).
    std::priority_queue<std::pair<float, uint32_t>> result;
    while (!top_candidates.empty()) {
        auto p = top_candidates.top();
        top_candidates.pop();
        result.push({p.first,
                     static_cast<uint32_t>(appr_alg->getExternalLabel(p.second))});
    }
    return result;
}
