// =============================================================================
// ivf_pq_simd.h — IVF + Product Quantization, two orderings
//
// Two ANNS algorithms that combine an Inverted File Index (IVF) with Product
// Quantization (PQ) compression, differing only in which component is applied
// first during the build phase:
//
//  IVF-first  (ivfpq_*):
//    Build: IVF k-means → compute residuals r_i = x_i − centroid[c_i] →
//           train PQ on residuals → encode residuals → store per-cluster codes.
//    Query: coarse centroid rank → for each selected cluster c:
//             q_res = query − centroid[c]
//             LUT[m][k] = IP(q_res_m, pq_centroid[m][k])
//             approx_ip(j) = IP(centroid[c], query) + Σ_m LUT[m][code[j][m]]
//           gather top-p approx candidates → exact NEON IP rerank → top-k.
//    Pro:  tight residual distribution → lower PQ quantization error → better recall.
//    Con:  nprobe LUT builds per query (cheap: M×K×(dsub/4) NEON FMAs each).
//
//  PQ-first  (pqivf_*):
//    Build: train PQ on all original vectors globally → encode all vectors →
//           IVF k-means → distribute global codes into per-cluster lists.
//    Query: build ONE global LUT from query → coarse centroid rank →
//           ADC scan of selected clusters → top-p → exact rerank → top-k.
//    Pro:  single LUT build per query, simpler.
//    Con:  PQ trained on full-range data → higher quantization error → lower recall.
//
// Both variants use:
//   - NEON vmlaq_f32 flat-SIMD for LUT construction (dsub % 4 == 0 required)
//   - simd_inner_product_neon_unroll for coarse centroid ranking
//   - simd_inner_product_neon_unroll for exact float32 reranking
//   - Scalar ADC scan (M table lookups + accumulate per code)
//
// Default parameters for DEEP100K (dim=96, normalized):
//   M=8   → dsub=12 dims per subspace (12 % 4 == 0 ✓)
//   K=256 → 8-bit codes (uint8_t)
//
// Shared index struct IVFPQIndex — both build functions populate the same
// fields; the per-cluster codes differ only in what they represent (residual
// codes for ivfpq_build, original-vector codes for pqivf_build).
//
// Platform: AArch64.
// Compile:  g++ main.cc -o main -O2 -fopenmp -lpthread -std=c++11
// =============================================================================
#pragma once
#include <sys/time.h>
#include <queue>
#include <utility>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <cstdio>
#include "ivf_flat_simd.h"               // ivf_kmeans, simd_inner_product_neon_unroll
#include "../Alg_normal/pq_flat_normal.h" // pq_kmeans


// =============================================================================
// IVFPQIndex — index structure shared by both orderings
//
// For IVF-first:  codes[c][j*M .. (j+1)*M) encodes the residual of the j-th
//                 vector in cluster c against pq_centroids trained on residuals.
// For PQ-first:   codes[c][j*M .. (j+1)*M) encodes the original j-th vector
//                 in cluster c against pq_centroids trained on full vectors.
// =============================================================================
struct IVFPQIndex {
    size_t nlist, vecdim, base_number;
    size_t M, K, dsub;

    // IVF part
    std::vector<float> ivf_centroids;  // [nlist × vecdim]

    // PQ part (meaning depends on which build was used)
    std::vector<float> pq_centroids;   // [M × K × dsub]

    // Per-cluster lists (both indexed the same way)
    std::vector<std::vector<uint32_t>> invlists;  // invlists[c] = original base IDs
    std::vector<std::vector<uint8_t>>  codes;      // codes[c] = M bytes per entry
};


// =============================================================================
// Internal helpers
// =============================================================================

// _ivfpq_lut_flat_simd — NEON flat-SIMD LUT construction
//
// Computes dtable[m*K + c] = IP(q_m, pq_centroid[m][c]) for all (m, c).
// q may be either the raw query (PQ-first) or a residual query (IVF-first).
// dsub must be a multiple of 4 (DEEP100K: dsub=12 ✓).
static inline void _ivfpq_lut_flat_simd(const IVFPQIndex& idx,
                                         const float* q, float* dtable)
{
    const size_t M    = idx.M;
    const size_t K    = idx.K;
    const size_t dsub = idx.dsub;

    for (size_t m = 0; m < M; ++m) {
        const float* q_m    = q + m * dsub;
        const float* c_base = idx.pq_centroids.data() + m * K * dsub;
        float*       row    = dtable + m * K;
        for (size_t c = 0; c < K; ++c) {
            const float* cv  = c_base + c * dsub;
            float32x4_t  sum = vdupq_n_f32(0.0f);
            for (size_t j = 0; j < dsub; j += 4)
                sum = vmlaq_f32(sum, vld1q_f32(q_m + j), vld1q_f32(cv + j));
            row[c] = vaddvq_f32(sum);
        }
    }
}

// _ivfpq_rerank — drain coarse_heap, rerank top-p candidates with exact NEON IP
static inline std::priority_queue<std::pair<float, uint32_t>>
_ivfpq_rerank(std::priority_queue<std::pair<float, uint32_t>>& coarse_heap,
              const float* base, const float* query, size_t k, size_t d)
{
    std::vector<uint32_t> cands;
    cands.reserve(coarse_heap.size());
    while (!coarse_heap.empty()) {
        cands.push_back(coarse_heap.top().second);
        coarse_heap.pop();
    }
    std::priority_queue<std::pair<float, uint32_t>> result;
    for (uint32_t id : cands) {
        float ip  = simd_inner_product_neon_unroll(base + (size_t)id * d, query, d);
        float dis = 1.0f - ip;
        if (result.size() < k)               { result.push({dis, id}); }
        else if (dis < result.top().first)   { result.push({dis, id}); result.pop(); }
    }
    return result;
}

// _ivfpq_encode_vec — encode one sub-vector slice against a PQ codebook row
static inline uint8_t _ivfpq_encode_sub(const float* v_m, const float* cents_m,
                                         size_t K, size_t dsub)
{
    float   best = std::numeric_limits<float>::max();
    uint8_t bk   = 0;
    for (size_t k = 0; k < K; ++k) {
        const float* cv = cents_m + k * dsub;
        float d2 = 0.0f;
        for (size_t j = 0; j < dsub; ++j) { float diff = v_m[j] - cv[j]; d2 += diff * diff; }
        if (d2 < best) { best = d2; bk = static_cast<uint8_t>(k); }
    }
    return bk;
}


// =============================================================================
// ivfpq_build — IVF-first build (PQ trained on residuals)
//
// Parameters:
//   nlist   : number of IVF clusters (typical: 256–1024; default 256)
//   M       : PQ subspaces (default 8); must divide vecdim evenly
//   K       : PQ centroids per subspace (default 256; max 256 for uint8)
//   n_iters : k-means iterations for both IVF and PQ (default 25)
// =============================================================================
void ivfpq_build(IVFPQIndex& idx, const float* base,
                 size_t n, size_t d,
                 size_t nlist  = 256,
                 size_t M      = 8,
                 size_t K      = 256,
                 int    n_iters = 25)
{
    idx.nlist       = nlist;
    idx.vecdim      = d;
    idx.base_number = n;
    idx.M           = M;
    idx.K           = K;
    idx.dsub        = d / M;
    const size_t dsub = idx.dsub;

    // ── Step 1: IVF clustering ────────────────────────────────────────────────
    idx.ivf_centroids.resize(nlist * d);
    idx.invlists.resize(nlist);

    std::vector<uint32_t> assign;
    ivf_kmeans(base, n, d, nlist, n_iters, idx.ivf_centroids.data(), assign);

    for (size_t i = 0; i < n; ++i)
        idx.invlists[assign[i]].push_back(static_cast<uint32_t>(i));

    // ── Step 2: Compute residuals r_i = x_i − centroid[c_i] ─────────────────
    std::vector<float> residuals(n * d);
    for (size_t i = 0; i < n; ++i) {
        const float* v = base      + i * d;
        const float* c = idx.ivf_centroids.data() + assign[i] * d;
        float*       r = residuals.data() + i * d;
        for (size_t j = 0; j < d; ++j) r[j] = v[j] - c[j];
    }

    // ── Step 3: Train PQ codebooks on residuals ───────────────────────────────
    idx.pq_centroids.resize(M * K * dsub, 0.0f);
    std::vector<float> sub_data(n * dsub);

    for (size_t m = 0; m < M; ++m) {
        for (size_t i = 0; i < n; ++i)
            memcpy(sub_data.data() + i * dsub,
                   residuals.data() + i * d + m * dsub,
                   dsub * sizeof(float));
        float* cents_m = idx.pq_centroids.data() + m * K * dsub;
        pq_kmeans(sub_data.data(), n, dsub, K, n_iters, cents_m);
    }

    // ── Step 4: Encode residuals, store per-cluster ───────────────────────────
    idx.codes.resize(nlist);
    for (size_t c = 0; c < nlist; ++c)
        idx.codes[c].resize(idx.invlists[c].size() * M);

    std::vector<size_t> pos(nlist, 0);  // next write index per cluster
    for (size_t i = 0; i < n; ++i) {
        uint32_t c     = assign[i];
        uint8_t* code  = idx.codes[c].data() + pos[c]++ * M;
        const float* r = residuals.data() + i * d;
        for (size_t m = 0; m < M; ++m)
            code[m] = _ivfpq_encode_sub(r + m * dsub,
                                        idx.pq_centroids.data() + m * K * dsub,
                                        K, dsub);
    }
}


// =============================================================================
// pqivf_build — PQ-first build (PQ trained on original vectors globally)
//
// Parameters: same meaning as ivfpq_build.
// =============================================================================
void pqivf_build(IVFPQIndex& idx, const float* base,
                 size_t n, size_t d,
                 size_t nlist   = 256,
                 size_t M       = 8,
                 size_t K       = 256,
                 int    n_iters = 25)
{
    idx.nlist       = nlist;
    idx.vecdim      = d;
    idx.base_number = n;
    idx.M           = M;
    idx.K           = K;
    idx.dsub        = d / M;
    const size_t dsub = idx.dsub;

    // ── Step 1: Train PQ on original vectors globally ─────────────────────────
    idx.pq_centroids.resize(M * K * dsub, 0.0f);
    std::vector<float> sub_data(n * dsub);

    for (size_t m = 0; m < M; ++m) {
        for (size_t i = 0; i < n; ++i)
            memcpy(sub_data.data() + i * dsub,
                   base + i * d + m * dsub,
                   dsub * sizeof(float));
        float* cents_m = idx.pq_centroids.data() + m * K * dsub;
        pq_kmeans(sub_data.data(), n, dsub, K, n_iters, cents_m);
    }

    // ── Step 2: Encode all base vectors ──────────────────────────────────────
    std::vector<uint8_t> global_codes(n * M);
    for (size_t i = 0; i < n; ++i) {
        uint8_t* code = global_codes.data() + i * M;
        for (size_t m = 0; m < M; ++m)
            code[m] = _ivfpq_encode_sub(base + i * d + m * dsub,
                                        idx.pq_centroids.data() + m * K * dsub,
                                        K, dsub);
    }

    // ── Step 3: IVF clustering on original vectors ────────────────────────────
    idx.ivf_centroids.resize(nlist * d);
    idx.invlists.resize(nlist);

    std::vector<uint32_t> assign;
    ivf_kmeans(base, n, d, nlist, n_iters, idx.ivf_centroids.data(), assign);

    for (size_t i = 0; i < n; ++i)
        idx.invlists[assign[i]].push_back(static_cast<uint32_t>(i));

    // ── Step 4: Distribute global codes into per-cluster lists ───────────────
    idx.codes.resize(nlist);
    for (size_t c = 0; c < nlist; ++c)
        idx.codes[c].resize(idx.invlists[c].size() * M);

    std::vector<size_t> pos(nlist, 0);
    for (size_t i = 0; i < n; ++i) {
        uint32_t c = assign[i];
        memcpy(idx.codes[c].data() + pos[c]++ * M,
               global_codes.data() + i * M, M);
    }
}


// =============================================================================
// ivfpq_save / ivfpq_load — binary serialization (shared by both orderings)
//
// Binary format (little-endian, native):
//   [6 × uint64] nlist, vecdim, base_number, M, K, dsub
//   ivf_centroids : nlist × vecdim × float32
//   pq_centroids  : M × K × dsub × float32
//   for each cluster c:
//     list_size (uint64)
//     list_size × uint32_t  (original IDs)
//     list_size × M bytes   (PQ codes)
// =============================================================================

static bool ivfpq_save(const IVFPQIndex& idx, const char* path)
{
    FILE* f = fopen(path, "wb");
    if (!f) return false;

    uint64_t hdr[6] = {
        (uint64_t)idx.nlist, (uint64_t)idx.vecdim, (uint64_t)idx.base_number,
        (uint64_t)idx.M,     (uint64_t)idx.K,       (uint64_t)idx.dsub
    };
    fwrite(hdr, sizeof(uint64_t), 6, f);
    fwrite(idx.ivf_centroids.data(), sizeof(float), idx.nlist * idx.vecdim, f);
    fwrite(idx.pq_centroids.data(),  sizeof(float), idx.M * idx.K * idx.dsub, f);

    for (size_t c = 0; c < idx.nlist; ++c) {
        uint64_t sz = idx.invlists[c].size();
        fwrite(&sz, sizeof(uint64_t), 1, f);
        if (sz > 0) {
            fwrite(idx.invlists[c].data(), sizeof(uint32_t), sz, f);
            fwrite(idx.codes[c].data(),    sizeof(uint8_t),  sz * idx.M, f);
        }
    }

    fclose(f);
    return true;
}

static bool ivfpq_load(IVFPQIndex& idx, const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f) return false;

    uint64_t hdr[6];
    if (fread(hdr, sizeof(uint64_t), 6, f) != 6) { fclose(f); return false; }
    idx.nlist       = (size_t)hdr[0];
    idx.vecdim      = (size_t)hdr[1];
    idx.base_number = (size_t)hdr[2];
    idx.M           = (size_t)hdr[3];
    idx.K           = (size_t)hdr[4];
    idx.dsub        = (size_t)hdr[5];

    idx.ivf_centroids.resize(idx.nlist * idx.vecdim);
    fread(idx.ivf_centroids.data(), sizeof(float), idx.nlist * idx.vecdim, f);
    idx.pq_centroids.resize(idx.M * idx.K * idx.dsub);
    fread(idx.pq_centroids.data(),  sizeof(float), idx.M * idx.K * idx.dsub, f);

    idx.invlists.resize(idx.nlist);
    idx.codes.resize(idx.nlist);
    for (size_t c = 0; c < idx.nlist; ++c) {
        uint64_t sz = 0;
        fread(&sz, sizeof(uint64_t), 1, f);
        idx.invlists[c].resize((size_t)sz);
        idx.codes[c].resize((size_t)sz * idx.M);
        if (sz > 0) {
            fread(idx.invlists[c].data(), sizeof(uint32_t), sz, f);
            fread(idx.codes[c].data(),    sizeof(uint8_t),  sz * idx.M, f);
        }
    }

    fclose(f);
    return true;
}


// =============================================================================
// ivfpq_search_simd — IVF-first search (per-cluster residual LUT)
//
// For each of the top-nprobe selected clusters c:
//   q_res = query − ivf_centroid[c]          (residual query for cluster c)
//   LUT[m][k] = IP(q_res_m, pq_centroid[m][k])
//   approx_ip(j) = bias_c + Σ_m LUT[m][code[j][m]]
//     where bias_c = IP(ivf_centroid[c], query)  (constant offset per cluster)
//
// Gathers top-p candidates across all scanned clusters, then reranks with
// exact NEON inner product against original base vectors.
//
// Parameters:
//   k      : final top-k to return
//   nprobe : number of IVF clusters to scan (latency-recall knob)
//   p      : coarse candidate pool before rerank (e.g. COARSE_P=200)
// =============================================================================
std::priority_queue<std::pair<float, uint32_t>>
ivfpq_search_simd(const IVFPQIndex& idx, const float* base,
                  const float* query, size_t k, size_t nprobe, size_t p)
{
    const size_t d  = idx.vecdim;
    const size_t M  = idx.M;
    const size_t K  = idx.K;
    const size_t np = std::min(nprobe, idx.nlist);

    // ── Phase 1: Coarse centroid rank ─────────────────────────────────────────
    std::vector<std::pair<float, uint32_t>> coarse(idx.nlist);
    for (size_t c = 0; c < idx.nlist; ++c) {
        float ip = simd_inner_product_neon_unroll(
            idx.ivf_centroids.data() + c * d, query, d);
        coarse[c] = {1.0f - ip, static_cast<uint32_t>(c)};
    }
    std::partial_sort(coarse.begin(), coarse.begin() + np, coarse.end());

    // ── Phase 2: Per-cluster ADC scan with residual LUT ───────────────────────
    std::vector<float> dtable(M * K);
    std::vector<float> q_res(d);
    std::priority_queue<std::pair<float, uint32_t>> coarse_heap;

    for (size_t probe = 0; probe < np; ++probe) {
        uint32_t    c    = coarse[probe].second;
        float       bias = 1.0f - coarse[probe].first;  // = IP(centroid[c], query)

        // Compute query residual for this cluster
        const float* cent = idx.ivf_centroids.data() + c * d;
        for (size_t j = 0; j < d; ++j) q_res[j] = query[j] - cent[j];

        // Build LUT from residual query (NEON flat-SIMD)
        _ivfpq_lut_flat_simd(idx, q_res.data(), dtable.data());

        // ADC scan: approximate IP = centroid_bias + Σ_m dtable[m][code[m]]
        const auto&    ids  = idx.invlists[c];
        const uint8_t* cptr = idx.codes[c].data();
        for (size_t j = 0; j < ids.size(); ++j) {
            const uint8_t* code = cptr + j * M;
            float approx_ip = bias;
            for (size_t m = 0; m < M; ++m) approx_ip += dtable[m * K + code[m]];
            float dis = 1.0f - approx_ip;
            if (coarse_heap.size() < p)                   { coarse_heap.push({dis, ids[j]}); }
            else if (dis < coarse_heap.top().first)        { coarse_heap.push({dis, ids[j]}); coarse_heap.pop(); }
        }
    }

    // ── Phase 3: Exact NEON IP rerank ─────────────────────────────────────────
    return _ivfpq_rerank(coarse_heap, base, query, k, d);
}


// =============================================================================
// pqivf_search_simd — PQ-first search (single global LUT)
//
// Builds ONE LUT from the raw query, then performs ADC scan only over the
// vectors in the top-nprobe IVF clusters.  No residual computation needed.
//
// Parameters: same as ivfpq_search_simd.
// =============================================================================
std::priority_queue<std::pair<float, uint32_t>>
pqivf_search_simd(const IVFPQIndex& idx, const float* base,
                  const float* query, size_t k, size_t nprobe, size_t p)
{
    const size_t d  = idx.vecdim;
    const size_t M  = idx.M;
    const size_t K  = idx.K;
    const size_t np = std::min(nprobe, idx.nlist);

    // ── Phase 0: Build global LUT once ───────────────────────────────────────
    std::vector<float> dtable(M * K);
    _ivfpq_lut_flat_simd(idx, query, dtable.data());

    // ── Phase 1: Coarse centroid rank ─────────────────────────────────────────
    std::vector<std::pair<float, uint32_t>> coarse(idx.nlist);
    for (size_t c = 0; c < idx.nlist; ++c) {
        float ip = simd_inner_product_neon_unroll(
            idx.ivf_centroids.data() + c * d, query, d);
        coarse[c] = {1.0f - ip, static_cast<uint32_t>(c)};
    }
    std::partial_sort(coarse.begin(), coarse.begin() + np, coarse.end());

    // ── Phase 2: ADC scan over selected clusters ───────────────────────────────
    std::priority_queue<std::pair<float, uint32_t>> coarse_heap;

    for (size_t probe = 0; probe < np; ++probe) {
        uint32_t       c    = coarse[probe].second;
        const auto&    ids  = idx.invlists[c];
        const uint8_t* cptr = idx.codes[c].data();
        for (size_t j = 0; j < ids.size(); ++j) {
            const uint8_t* code = cptr + j * M;
            float approx_ip = 0.0f;
            for (size_t m = 0; m < M; ++m) approx_ip += dtable[m * K + code[m]];
            float dis = 1.0f - approx_ip;
            if (coarse_heap.size() < p)                   { coarse_heap.push({dis, ids[j]}); }
            else if (dis < coarse_heap.top().first)        { coarse_heap.push({dis, ids[j]}); coarse_heap.pop(); }
        }
    }

    // ── Phase 3: Exact NEON IP rerank ─────────────────────────────────────────
    return _ivfpq_rerank(coarse_heap, base, query, k, d);
}
