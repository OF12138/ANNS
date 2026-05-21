// =============================================================================
// ivf_flat_simd.h — IVF-SIMD baseline: Inverted File Index + NEON search
//
// Algorithm overview:
//   Build:  k-means clusters base data into nlist clusters.
//           Store each cluster's centroid and the list of original vector
//           indices (inverted list) that belong to it.
//           Optional memory reordering: copies cluster vectors contiguously
//           so that the fine-scan phase reads sequential memory instead of
//           doing random jumps into the 38 MB base array.
//
//   Query (two phases):
//     1. Coarse: compute IP(query, centroid[c]) for all nlist centroids
//                using simd_inner_product_neon_unroll.  Select top-nprobe
//                clusters by partial sort.
//     2. Fine:   for each selected cluster, compute exact float32 IP against
//                every vector in that cluster.  Maintain a max-heap of top-k.
//
//   nprobe is the latency-recall knob: larger nprobe → higher recall, more
//   latency.  Sweep nprobe to produce the recall-latency curve.
//
// Memory reordering (IVF_REORDER):
//   Without reorder: fine scan accesses base[invlists[c][j] * d] — random
//     jumps across the 38 MB base array per cluster → many cache misses.
//   With reorder: cluster c's vectors are stored contiguously in
//     reordered_base[cluster_offset[c]*d .. cluster_offset[c+1]*d) →
//     sequential reads, much higher cache hit rate for the fine scan.
//   Compile with IVF_REORDER=0 or 1 to compare the cache effect.
//
// Precondition: vecdim % 16 == 0  (DEEP100K: 96 % 16 = 0 ✓)
// Platform:    AArch64.
// Compile:     g++ main.cc -o main -O2 -fopenmp -lpthread -std=c++11
// =============================================================================
#pragma once
#include <arm_neon.h>
#include <queue>
#include <utility>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include "flat_simd.h"   // simd_inner_product_neon_unroll


// =============================================================================
// IVFIndex — index structure
// =============================================================================

struct IVFIndex 
{
    size_t nlist;        // number of clusters
    size_t vecdim;
    size_t base_number;
    bool   reordered;

    std::vector<float>    centroids;  // [nlist × vecdim]
    // invlists[c] = original base indices assigned to cluster c
    std::vector<std::vector<uint32_t>> invlists;

    // Populated only when reordered=true:
    //   reordered_base[cluster_offset[c]*vecdim .. cluster_offset[c+1]*vecdim)
    //   holds the raw float vectors of cluster c in the same order as invlists[c].
    std::vector<float>  reordered_base;   // [base_number × vecdim]
    std::vector<size_t> cluster_offset;   // [nlist+1]
};


// =============================================================================
// ivf_kmeans — k-means clustering (NEON-accelerated assign step)
//
// Runs n_iters of Lloyd's algorithm.  The assign step uses
// simd_inner_product_neon_unroll for the IP part of the L2 distance:
//   d2(v, c_k) = ||v||^2 + ||c_k||^2 - 2 * IP(v, c_k)
//
// assign[i] is set to the nearest centroid index for base vector i.
// centroids[0..K*d) is updated in-place; caller must pre-allocate.
// =============================================================================
static void ivf_kmeans(const float* data, size_t n, size_t d,
                       size_t K, int n_iters,
                       float* centroids, std::vector<uint32_t>& assign)
{
    // LCG-shuffle: pick K distinct vectors as initial centroids
    std::vector<size_t> perm(n);
    for (size_t i = 0; i < n; ++i) perm[i] = i;
    unsigned rng = 0xA5A5A5A5u;
    for (size_t i = n - 1; i > 0; --i) {
        rng = rng * 1664525u + 1013904223u;
        size_t j = rng % (i + 1);
        std::swap(perm[i], perm[j]);
    }
    for (size_t k = 0; k < K; ++k)
        memcpy(centroids + k * d, data + perm[k] * d, d * sizeof(float));

    assign.resize(n);
    std::vector<float> norm2_c(K);
    std::vector<float> cnt(K);

    for (int iter = 0; iter < n_iters; ++iter) {
        // Precompute ||centroid_k||^2 for all k
        for (size_t k = 0; k < K; ++k) {
            const float* c = centroids + k * d;
            float n2 = 0.0f;
            for (size_t j = 0; j < d; ++j) n2 += c[j] * c[j];
            norm2_c[k] = n2;
        }

        // Assign: find nearest centroid for each vector
        for (size_t i = 0; i < n; ++i) {
            const float* v = data + i * d;
            // ||v||^2
            float norm2_v = 0.0f;
            for (size_t j = 0; j < d; ++j) norm2_v += v[j] * v[j];

            float    best_d2 = std::numeric_limits<float>::max();
            uint32_t best_k  = 0;
            for (size_t k = 0; k < K; ++k) {
                float ip = simd_inner_product_neon_unroll(v, centroids + k * d, d);
                float d2 = norm2_v + norm2_c[k] - 2.0f * ip;
                if (d2 < best_d2) { best_d2 = d2; best_k = static_cast<uint32_t>(k); }
            }
            assign[i] = best_k;
        }

        // Update: recompute centroids as cluster means
        memset(centroids, 0, K * d * sizeof(float));
        std::fill(cnt.begin(), cnt.end(), 0.0f);
        for (size_t i = 0; i < n; ++i) {
            const float* v = data + i * d;
            float*       c = centroids + assign[i] * d;
            for (size_t j = 0; j < d; ++j) c[j] += v[j];
            cnt[assign[i]] += 1.0f;
        }
        for (size_t k = 0; k < K; ++k) {
            if (cnt[k] > 0.0f) {
                float  inv = 1.0f / cnt[k];
                float* c   = centroids + k * d;
                for (size_t j = 0; j < d; ++j) c[j] *= inv;
            }
        }
    }
}


// =============================================================================
// ivf_build — build IVFIndex from base dataset
//
// Parameters:
//   base     : base dataset [n × d]
//   n        : number of base vectors
//   d        : vector dimension (must satisfy d % 16 == 0)
//   nlist    : number of clusters (typical: 64–4096; default 256)
//   n_iters  : k-means iterations (default 25)
//   reorder  : if true, copy cluster vectors contiguously for cache locality
// =============================================================================
void ivf_build(IVFIndex& idx, const float* base, size_t n, size_t d,
               size_t nlist = 256, int n_iters = 25, bool reorder = false)
{
    idx.nlist        = nlist;
    idx.vecdim       = d;
    idx.base_number  = n;
    idx.reordered    = reorder;

    idx.centroids.resize(nlist * d);
    idx.invlists.resize(nlist);

    std::vector<uint32_t> assign;
    ivf_kmeans(base, n, d, nlist, n_iters, idx.centroids.data(), assign);

    // Build inverted lists
    for (size_t i = 0; i < n; ++i)
        idx.invlists[assign[i]].push_back(static_cast<uint32_t>(i));

    if (reorder) 
    {
        // cluster_offset[c] = number of vectors before cluster c in reordered_base
        idx.cluster_offset.resize(nlist + 1, 0);
        for (size_t c = 0; c < nlist; ++c)
            idx.cluster_offset[c + 1] = idx.cluster_offset[c] + idx.invlists[c].size();

        // Copy vectors in cluster order: cluster 0 first, then 1, ...
        // reordered_base[cluster_offset[c]+j] = base[invlists[c][j]]
        idx.reordered_base.resize(n * d);
        for (size_t c = 0; c < nlist; ++c) {
            size_t pos = idx.cluster_offset[c];
            for (uint32_t orig : idx.invlists[c]) {
                memcpy(idx.reordered_base.data() + pos * d,
                       base + (size_t)orig * d,
                       d * sizeof(float));
                ++pos;
            }
        }
    }
}


// =============================================================================
// ivf_search_simd — two-phase IVF query (coarse + fine), NEON inner product
//
// Parameters:
//   idx    : built IVFIndex
//   base   : original base dataset (used only when idx.reordered == false)
//   query  : single query vector [vecdim]
//   k      : top-k results to return
//   nprobe : number of clusters to scan in fine phase (latency-recall knob)
//
// Returns: max-heap of (IP distance, original base index) pairs.
// =============================================================================
std::priority_queue<std::pair<float, uint32_t>>
ivf_search_simd(const IVFIndex& idx, const float* base,
                const float* query, size_t k, size_t nprobe)
{
    const size_t d  = idx.vecdim;
    const size_t np = std::min(nprobe, idx.nlist);

    // ── Phase 1: Coarse — rank all nlist centroids by IP distance ─────────────
    std::vector<std::pair<float, uint32_t>> coarse(idx.nlist);
    for (size_t c = 0; c < idx.nlist; ++c) {
        float ip = simd_inner_product_neon_unroll(
            idx.centroids.data() + c * d, query, d);
        coarse[c] = {1.0f - ip, static_cast<uint32_t>(c)};
    }
    // Bring top-np (smallest distance) to front; remaining order unimportant
    std::partial_sort(coarse.begin(), coarse.begin() + np, coarse.end());

    // ── Phase 2: Fine — exact NEON IP scan over the top-np clusters ───────────
    std::priority_queue<std::pair<float, uint32_t>> heap;

    if (idx.reordered) {
        // Vectors are stored contiguously per cluster → sequential memory reads
        for (size_t probe = 0; probe < np; ++probe) {
            uint32_t c     = coarse[probe].second;
            size_t   start = idx.cluster_offset[c];
            size_t   end   = idx.cluster_offset[c + 1];
            for (size_t j = start; j < end; ++j) {
                float ip  = simd_inner_product_neon_unroll(
                    idx.reordered_base.data() + j * d, query, d);
                float dis = 1.0f - ip;
                uint32_t orig = idx.invlists[c][j - start];
                if (heap.size() < k) {
                    heap.push({dis, orig});
                } else if (dis < heap.top().first) {
                    heap.push({dis, orig});
                    heap.pop();
                }
            }
        }
    } else {
        // Random access into original base array via invlists[c][j] indices
        for (size_t probe = 0; probe < np; ++probe) {
            uint32_t c = coarse[probe].second;
            for (uint32_t orig : idx.invlists[c]) {
                float ip  = simd_inner_product_neon_unroll(
                    base + (size_t)orig * d, query, d);
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
    return heap;
}
