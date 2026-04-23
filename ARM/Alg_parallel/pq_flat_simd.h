// =============================================================================
// pq_flat_simd.h — NEON-accelerated PQ LUT construction + two-phase k-NN search
//
// Implements two SIMD strategies for the ADC distance-table (LUT) build step,
// which dominates per-query overhead when the codebook is large (K=256, M=8).
//
// ── Strategy 1: Flat-SIMD ────────────────────────────────────────────────────
// For each centroid c in subspace m, compute the dsub-dim IP against q_m using
// NEON vmlaq_f32 (4 muls/FMA per instruction).  Identical structure to
// sq8_coarse_dot_simd but over dsub dims instead of vecdim.
//
//   For each (m, c):
//     sum ← Σ_{j=0}^{dsub-1/4} q_m[j:j+4] ⊙ centroid[m][c][j:j+4]
//     dtable[m][c] = vaddvq_f32(sum)
//
//   FMA count:   M × K × (dsub/4)  = 8×256×3 = 6144 vector FMAs
//   Reductions:  M × K              = 2048 (one vaddvq per centroid)
//
// ── Strategy 2: Cross-centroid SIMD ─────────────────────────────────────────
// Precompute a transposed centroid array cents_t[m][j][c] (all K centroids for
// subspace m, dim j are contiguous).  Then process 4 centroids at once:
//
//   For each (m, c_group of 4):
//     acc[0..3] ← Σ_j  q_m[j] ⊗ cents_t[m][j][c:c+4]   (vmlaq_n_f32)
//     dtable[m][c:c+4] ← vst1q_f32(acc)
//
//   The scalar-broadcast FMA vmlaq_n_f32(acc, cv4, q_m[j]) multiplies all 4
//   centroid values by the same query scalar in one instruction — no explicit
//   broadcast needed, and the result for all 4 centroids accumulates in one
//   register.  No horizontal reduction required (vst1q_f32 writes 4 results).
//
//   FMA count:   M × (K/4) × dsub   = 8×64×12 = 6144 SIMD FMAs
//   Reductions:  0  (eliminated entirely)
//   Memory:      transposed layout makes the K-stride inner access sequential
//
// PQIndexSIMD wraps PQIndex and precomputes the transposed codebook once so
// that the cross-centroid LUT can be rebuilt each query without re-transposing.
//
// Preconditions:
//   Flat-SIMD:           dsub % 4 == 0   (96/8 = 12 → 12%4 = 0 ✓)
//   Cross-centroid SIMD: K    % 4 == 0   (256  % 4 = 0 ✓), any dsub
// =============================================================================
#pragma once
#include <arm_neon.h>
#include <queue>
#include <utility>
#include <vector>
#include <cstdint>
#include "../Alg_normal/pq_flat_normal.h"

// ---------------------------------------------------------------------------
// PQIndexSIMD — PQIndex + transposed centroid layout for cross-centroid SIMD
//
// cents_t[m * dsub * K + j * K + c]:  subspace m, dim j, centroid c
// Transposing from [m][c][j] → [m][j][c] makes all K values for a given
// (subspace m, dim j) contiguous, enabling a sequential vld1q_f32 load of
// 4 centroid values at once in pq_build_lut_cross_centroid_simd().
// ---------------------------------------------------------------------------
struct PQIndexSIMD 
{
    const PQIndex* idx;
    std::vector<float> cents_t;  // transposed: M * dsub * K floats

    explicit PQIndexSIMD(const PQIndex& base_idx) : idx(&base_idx)
    {
        const size_t M    = base_idx.M;
        const size_t K    = base_idx.K;
        const size_t dsub = base_idx.dsub;
        cents_t.resize(M * dsub * K);

        for (size_t m = 0; m < M; ++m) 
        {
            const float* src = base_idx.centroids.data() + m * K * dsub; // [c][j]
            float*       dst = cents_t.data()            + m * dsub * K; // [j][c]
            for (size_t c = 0; c < K; ++c)
                for (size_t j = 0; j < dsub; ++j)
                    dst[j * K + c] = src[c * dsub + j];
        }
    }
};

// ---------------------------------------------------------------------------
// pq_build_lut_flat_simd — Flat-SIMD LUT construction
//
// Replaces the scalar dsub-dim inner product in the LUT loop with a NEON
// vmlaq_f32 kernel: dsub/4 iterations × 4 FMAs per iteration.
// One vaddvq_f32 horizontal reduction per centroid (M×K total reductions).
// ---------------------------------------------------------------------------
inline void pq_build_lut_flat_simd(const PQIndex& index, const float* query,
                                    float* dtable)
{
    const size_t M    = index.M;
    const size_t K    = index.K;
    const size_t dsub = index.dsub;

    for (size_t m = 0; m < M; ++m) 
    {
        const float* q_m    = query + m * dsub;
        const float* c_base = index.centroids.data() + m * K * dsub;
        float*       row    = dtable + m * K;

        for (size_t c = 0; c < K; ++c) 
        {
            const float* cv   = c_base + c * dsub;
            float32x4_t  sum  = vdupq_n_f32(0.0f);
            for (size_t j = 0; j < dsub; j += 4) sum = vmlaq_f32(sum, vld1q_f32(q_m + j), vld1q_f32(cv + j));
            row[c] = vaddvq_f32(sum);   // horizontal reduce 4 lanes → scalar
        }
    }
}

// ---------------------------------------------------------------------------
// pq_build_lut_cross_centroid_simd — cross-centroid SIMD LUT construction
//
// Processes 4 centroids simultaneously using the transposed layout.
// For each dim j, broadcasts q_m[j] as a scalar and multiplies against
// 4 consecutive centroid values loaded from cents_t[m][j][c:c+4].
// vmlaq_n_f32(acc, cv4, q_m[j]) = acc + cv4 * q_m[j]  (scalar broadcast FMA)
// After dsub iterations, acc holds 4 complete IP values; vst1q_f32 stores all.
// No horizontal reduction needed — 2048 reductions eliminated vs. flat SIMD.
// ---------------------------------------------------------------------------
inline void pq_build_lut_cross_centroid_simd(const PQIndexSIMD& pq_simd,
                                              const float* query, float* dtable)
{
    const PQIndex& index = *pq_simd.idx;
    const size_t M    = index.M;
    const size_t K    = index.K;
    const size_t dsub = index.dsub;

    for (size_t m = 0; m < M; ++m) 
    {
        const float* q_m = query + m * dsub;
        const float* t_m = pq_simd.cents_t.data() + m * dsub * K; // [j][c]
        float*       row = dtable + m * K;

        // Process K centroids in groups of 4; compute 4 IPs per iteration
        for (size_t c = 0; c < K; c += 4) 
        {
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (size_t j = 0; j < dsub; ++j) 
            {
                // Load 4 centroid values at dim j (sequential in transposed layout)
                float32x4_t cv4 = vld1q_f32(t_m + j * K + c);
                // Scalar broadcast FMA: acc[i] += cv4[i] * q_m[j]
                acc = vmlaq_n_f32(acc, cv4, q_m[j]);
            }
            vst1q_f32(row + c, acc);  // store 4 IP values at once
        }
    }
}

// ---------------------------------------------------------------------------
// Shared Phase-1 + Phase-2 scan, templated on LUT build type.
// Phase 1: coarse ADC scan → top-p by approximate IP.
// Phase 2: exact float32 IP rerank → top-k.
// ---------------------------------------------------------------------------
static inline std::priority_queue<std::pair<float, uint32_t>>
pq_rerank_from_dtable(const PQIndex& index, const float* base,
                      const float* query, size_t k, size_t p,
                      const float* dtable)
{
    const size_t M = index.M;
    const size_t K = index.K;
    const size_t d = index.vecdim;

    std::priority_queue<std::pair<float, uint32_t>> coarse_heap;
    for (size_t i = 0; i < index.base_number; ++i) 
    {
        const uint8_t* code = index.codes.data() + i * M;
        float approx_ip = 0.0f;
        for (size_t m = 0; m < M; ++m) approx_ip += dtable[m * K + code[m]];
        float dis = 1.0f - approx_ip;

        if (coarse_heap.size() < p) 
        {
            coarse_heap.push({dis, static_cast<uint32_t>(i)});
        } 
        else if (dis < coarse_heap.top().first) 
        {
            coarse_heap.push({dis, static_cast<uint32_t>(i)});
            coarse_heap.pop();
        }
    }

    std::vector<uint32_t> cand_ids;
    cand_ids.reserve(p);
    while (!coarse_heap.empty()) {
        cand_ids.push_back(coarse_heap.top().second);
        coarse_heap.pop();
    }

    std::priority_queue<std::pair<float, uint32_t>> result;
    for (uint32_t id : cand_ids) {
        const float* bv = base + static_cast<size_t>(id) * d;
        float ip = 0.0f;
        for (size_t j = 0; j < d; ++j) ip += bv[j] * query[j];
        float dis = 1.0f - ip;

        if (result.size() < k) {
            result.push({dis, id});
        } else if (dis < result.top().first) {
            result.push({dis, id});
            result.pop();
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// pq_flat_search_rerank_flat_simd — two-phase PQ search, Flat-SIMD LUT
// ---------------------------------------------------------------------------
inline std::priority_queue<std::pair<float, uint32_t>>
pq_flat_search_rerank_flat_simd(const PQIndex& index, const float* base,
                                 const float* query, size_t k, size_t p)
{
    std::vector<float> dtable(index.M * index.K);
    pq_build_lut_flat_simd(index, query, dtable.data());
    return pq_rerank_from_dtable(index, base, query, k, p, dtable.data());
}

// ---------------------------------------------------------------------------
// pq_flat_search_rerank_cross_centroid_simd — two-phase PQ search, cross-centroid LUT
// ---------------------------------------------------------------------------
inline std::priority_queue<std::pair<float, uint32_t>>
pq_flat_search_rerank_cross_centroid_simd(const PQIndexSIMD& pq_simd,
                                           const float* base, const float* query,
                                           size_t k, size_t p)
{
    const PQIndex& index = *pq_simd.idx;
    std::vector<float> dtable(index.M * index.K);
    pq_build_lut_cross_centroid_simd(pq_simd, query, dtable.data());
    return pq_rerank_from_dtable(index, base, query, k, p, dtable.data());
}

// =============================================================================
// Cross-centroid LUT variant: 4× loop unrolling
//
// Processes 16 centroids per outer iteration via 4 independent acc registers.
// Four independent float32x4_t accumulators (acc0..acc3) break the FMA
// dependency chain: vmlaq_n_f32 latency ≈ 4 cycles on AArch64, but throughput
// is 1/cycle, so 4 independent chains keep all FMA slots busy.
// q_m[j] is loaded once per j and broadcast across all 4 accumulator updates.
// Precondition: K % 16 == 0  (256 % 16 = 0 ✓)
// =============================================================================

// ---------------------------------------------------------------------------
// pq_build_lut_cc_unroll — cross-centroid + 4× loop unrolling
// ---------------------------------------------------------------------------
inline void pq_build_lut_cc_unroll(const PQIndexSIMD& pq_simd,
                                    const float* query, float* dtable)
{
    const PQIndex& index = *pq_simd.idx;
    const size_t M    = index.M;
    const size_t K    = index.K;
    const size_t dsub = index.dsub;

    for (size_t m = 0; m < M; ++m) {
        const float* q_m = query + m * dsub;
        const float* t_m = pq_simd.cents_t.data() + m * dsub * K;
        float*       row = dtable + m * K;

        for (size_t c = 0; c < K; c += 16) {
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);
            for (size_t j = 0; j < dsub; ++j) {
                const float        q_j = q_m[j];
                const float* const ptr = t_m + j * K + c;
                acc0 = vmlaq_n_f32(acc0, vld1q_f32(ptr),      q_j);
                acc1 = vmlaq_n_f32(acc1, vld1q_f32(ptr +  4), q_j);
                acc2 = vmlaq_n_f32(acc2, vld1q_f32(ptr +  8), q_j);
                acc3 = vmlaq_n_f32(acc3, vld1q_f32(ptr + 12), q_j);
            }
            vst1q_f32(row + c,      acc0);
            vst1q_f32(row + c +  4, acc1);
            vst1q_f32(row + c +  8, acc2);
            vst1q_f32(row + c + 12, acc3);
        }
    }
}

// ---------------------------------------------------------------------------
// pq_flat_search_rerank_cc_unroll — two-phase PQ search, unrolled LUT
// ---------------------------------------------------------------------------
inline std::priority_queue<std::pair<float, uint32_t>>
pq_flat_search_rerank_cc_unroll(const PQIndexSIMD& pq_simd, const float* base,
                                 const float* query, size_t k, size_t p)
{
    const PQIndex& index = *pq_simd.idx;
    std::vector<float> dtable(index.M * index.K);
    pq_build_lut_cc_unroll(pq_simd, query, dtable.data());
    return pq_rerank_from_dtable(index, base, query, k, p, dtable.data());
}

// =============================================================================
// SIMD-accelerated k-means index building
//
// The k-means assign step dominates build time: for each of n vectors, compute
// L2 distance to all K centroids over dsub dims.
//
//   d2(v, c_k) = ||v||^2 + ||c_k||^2 - 2 * IP(v, c_k)
//
// ||v||^2 is computed once per vector; ||c_k||^2 is precomputed per iteration.
// IP(v, c_k) for all K centroids is computed via cross-centroid SIMD:
//   - Transpose centroids to cents_t[j][k] once per iteration.
//   - For each vector, compute K IPs with the same blocked + unrolled kernel
//     used in pq_build_lut_cc_unroll (treating the vector like a 1-element query).
//
// Cache blocking (BLOCK_K=64): tile = dsub × BLOCK_K × 4 bytes = 3 KB.
// Across n=100K vectors the centroid tile stays hot in L1 for all vectors in a
// block before eviction, amortising the initial cold-miss cost of each tile.
// 4× unrolling: 4 independent acc registers hide the ~4-cycle FMA latency.
//
// Preconditions: K % 16 == 0, BLOCK_K % 16 == 0  (both satisfied for K=256)
// =============================================================================

static constexpr size_t PQ_KMEANS_BLOCK_K = 64;
static constexpr size_t PQ_KMEANS_BLOCK_N = 128; // vectors per i-tile for true matrix-matrix blocking

// // pq_kmeans_simd — drop-in replacement for pq_kmeans using NEON cross-centroid
// static inline void pq_kmeans_simd(const float* data, size_t n, size_t dsub,
//                                    size_t K, int n_iters, float* centroids)
// {
//     // LCG-shuffle initialization (identical to pq_kmeans)
//     std::vector<size_t> perm(n);
//     for (size_t i = 0; i < n; ++i) perm[i] = i;
//     unsigned rng = 0xA5A5A5A5u;
//     for (size_t i = n - 1; i > 0; --i)
//     {
//         rng = rng * 1664525u + 1013904223u;
//         size_t j = rng % (i + 1);
//         size_t tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
//     }
//     for (size_t k = 0; k < K; ++k)
//         memcpy(centroids + k * dsub, data + perm[k] * dsub, dsub * sizeof(float));

//     std::vector<uint32_t> assign(n);
//     std::vector<float>    cnt(K);
//     std::vector<float>    cents_t(dsub * K);  // transposed layout [j][k]
//     std::vector<float>    norm2_c(K);
//     std::vector<float>    ip_buf(K);

//     for (int iter = 0; iter < n_iters; ++iter)
//     {
//         // Transpose centroids [k][j] → [j][k] and precompute ||c_k||^2
//         for (size_t k = 0; k < K; ++k)
//         {
//             float n2 = 0.0f;
//             for (size_t j = 0; j < dsub; ++j)
//             {
//                 float v = centroids[k * dsub + j];
//                 cents_t[j * K + k] = v;
//                 n2 += v * v;
//             }
//             norm2_c[k] = n2;
//         }

//         // Assign step: cross-centroid blocked + unrolled SIMD
//         for (size_t i = 0; i < n; ++i)
//         {
//             const float* v = data + i * dsub;

//             float norm2_v = 0.0f;
//             for (size_t j = 0; j < dsub; ++j) norm2_v += v[j] * v[j];

//             // Compute IP(v, all K centroids) with BLOCK_K tiling + 4× unroll
//             for (size_t k_blk = 0; k_blk < K; k_blk += PQ_KMEANS_BLOCK_K)
//             {
//                 for (size_t k = k_blk; k < k_blk + PQ_KMEANS_BLOCK_K; k += 16)
//                 {
//                     float32x4_t acc0 = vdupq_n_f32(0.0f);
//                     float32x4_t acc1 = vdupq_n_f32(0.0f);
//                     float32x4_t acc2 = vdupq_n_f32(0.0f);
//                     float32x4_t acc3 = vdupq_n_f32(0.0f);
//                     for (size_t j = 0; j < dsub; ++j)
//                     {
//                         const float        v_j = v[j];
//                         const float* const ptr = cents_t.data() + j * K + k;
//                         acc0 = vmlaq_n_f32(acc0, vld1q_f32(ptr),      v_j);
//                         acc1 = vmlaq_n_f32(acc1, vld1q_f32(ptr +  4), v_j);
//                         acc2 = vmlaq_n_f32(acc2, vld1q_f32(ptr +  8), v_j);
//                         acc3 = vmlaq_n_f32(acc3, vld1q_f32(ptr + 12), v_j);
//                     }
//                     vst1q_f32(ip_buf.data() + k,      acc0);
//                     vst1q_f32(ip_buf.data() + k +  4, acc1);
//                     vst1q_f32(ip_buf.data() + k +  8, acc2);
//                     vst1q_f32(ip_buf.data() + k + 12, acc3);
//                 }
//             }

//             // Find nearest centroid: argmin of d2 = norm2_v + norm2_c[k] - 2*ip[k]
//             float    best_d2 = std::numeric_limits<float>::max();
//             uint32_t best_k  = 0;
//             for (size_t k = 0; k < K; ++k)
//             {
//                 float d2 = norm2_v + norm2_c[k] - 2.0f * ip_buf[k];
//                 if (d2 < best_d2) { best_d2 = d2; best_k = static_cast<uint32_t>(k); }
//             }
//             assign[i] = best_k;
//         }

//         // Update step: recompute centroids as cluster means (scalar)
//         memset(centroids, 0, K * dsub * sizeof(float));
//         std::fill(cnt.begin(), cnt.end(), 0.0f);
//         for (size_t i = 0; i < n; ++i)
//         {
//             const float* v = data + i * dsub;
//             float*       c = centroids + assign[i] * dsub;
//             for (size_t j = 0; j < dsub; ++j) c[j] += v[j];
//             cnt[assign[i]] += 1.0f;
//         }
//         for (size_t k = 0; k < K; ++k)
//         {
//             if (cnt[k] > 0.0f)
//             {
//                 float inv = 1.0f / cnt[k];
//                 float* c  = centroids + k * dsub;
//                 for (size_t j = 0; j < dsub; ++j) c[j] *= inv;
//             }
//         }
//     }
// }

// // pq_build_index_simd — builds a PQIndex using pq_kmeans_simd instead of pq_kmeans
// // Same interface and default parameters as PQIndex::build; original is untouched.
// // NOTE: uses only k-tiling (BLOCK_K), not i-tiling. See pq_build_index_simd_blocked
// //       for true matrix-matrix double-tiled blocking.
// inline void pq_build_index_simd(PQIndex& out, const float* base, size_t n, size_t d,
//                                  size_t m_subs = 8, size_t k_cents = 256, int n_iters = 25)
// {
//     out.M           = m_subs;
//     out.K           = k_cents;
//     out.dsub        = d / m_subs;
//     out.vecdim      = d;
//     out.base_number = n;

//     out.centroids.resize(m_subs * k_cents * out.dsub, 0.0f);
//     out.codes.resize(n * m_subs);

//     std::vector<float> sub_data(n * out.dsub);

//     for (size_t m = 0; m < m_subs; ++m) {
//         for (size_t i = 0; i < n; ++i)
//             memcpy(sub_data.data() + i * out.dsub,
//                    base + i * d + m * out.dsub,
//                    out.dsub * sizeof(float));

//         float* cents_m = out.centroids.data() + m * k_cents * out.dsub;
//         pq_kmeans_simd(sub_data.data(), n, out.dsub, k_cents, n_iters, cents_m);

//         // Encode: assign each base vector to its nearest centroid
//         for (size_t i = 0; i < n; ++i) {
//             const float* v    = sub_data.data() + i * out.dsub;
//             float        best = std::numeric_limits<float>::max();
//             uint8_t      bk   = 0;
//             for (size_t k = 0; k < k_cents; ++k) {
//                 const float* c = cents_m + k * out.dsub;
//                 float d2 = 0.0f;
//                 for (size_t j = 0; j < out.dsub; ++j) {
//                     float diff = v[j] - c[j];
//                     d2 += diff * diff;
//                 }
//                 if (d2 < best) { best = d2; bk = static_cast<uint8_t>(k); }
//             }
//             out.codes[i * m_subs + m] = bk;
//         }
//     }
// }

// =============================================================================
// NEON gather coarse scan
//
// ARM has no hardware gather instruction for float32.  We emulate one with
// vld1_dup_f32 + vld1_lane_f32 + vcombine_f32, which generates exactly 4
// scalar loads and places them into one float32x4_t.  Calling neon_gather4
// twice covers all M=8 subspaces; vaddvq_f32(vaddq_f32(lo,hi)) reduces the
// 8 values to a single approx_ip in one tree of NEON adds.
//
// The outer loop is 4× unrolled so that the 32 gather loads for four
// consecutive base vectors are in-flight simultaneously, hiding the ~4-cycle
// load-use latency on Kunpeng-920's out-of-order engine.
//
// Precondition: index.M == 8  (DEEP100K: M=8, K=256 ✓)
// =============================================================================

// ---------------------------------------------------------------------------
// neon_gather4 — load 4 floats from 4 independent addresses into float32x4
//
// vld1_dup_f32(p)        : [*p, *p]            (broadcast)
// vld1_lane_f32(p,v,1)   : [v[0], *p]          (replace lane 1)
// vcombine_f32(lo, hi)   : [lo[0], lo[1], hi[0], hi[1]]
// Result                 : [*p0, *p1, *p2, *p3]
// ---------------------------------------------------------------------------
static inline float32x4_t neon_gather4(const float* p0, const float* p1,
                                        const float* p2, const float* p3)
{
    float32x2_t lo = vld1_dup_f32(p0);
    lo             = vld1_lane_f32(p1, lo, 1);
    float32x2_t hi = vld1_dup_f32(p2);
    hi             = vld1_lane_f32(p3, hi, 1);
    return vcombine_f32(lo, hi);
}

// ---------------------------------------------------------------------------
// pq_rerank_from_dtable_gather — coarse scan with NEON gather + rerank
//
// Drop-in replacement for pq_rerank_from_dtable.
// Inner loop change: scalar `for m: approx_ip += dtable[m*K+code[m]]`
// → two neon_gather4 calls + vaddvq_f32, eliminating the M=8 inner loop.
// Outer loop unrolled 4× to expose 32 independent gather loads per iteration.
// ---------------------------------------------------------------------------
static inline std::priority_queue<std::pair<float, uint32_t>>
pq_rerank_from_dtable_gather(const PQIndex& index, const float* base,
                              const float* query,  size_t k, size_t p,
                              const float* dtable)
{
    const size_t M = index.M;   // 8
    const size_t K = index.K;
    const size_t d = index.vecdim;
    const size_t N = index.base_number;

    // Per-subspace LUT base pointers — avoids m*K multiply inside hot loop
    const float* lut[8] = 
    {
        dtable,       dtable +   K, dtable + 2*K, dtable + 3*K,
        dtable + 4*K, dtable + 5*K, dtable + 6*K, dtable + 7*K
    };

    const uint8_t* codes = index.codes.data();

    std::priority_queue<std::pair<float, uint32_t>> coarse_heap;

    // Heap insertion helper (lambda keeps the hot loop free of duplicated code)
    auto heap_push = [&](float ip, uint32_t id) 
    {
        float dis = 1.0f - ip;
        if (coarse_heap.size() < p) 
        {
            coarse_heap.push({dis, id});
        } 
        else if (dis < coarse_heap.top().first) 
        {
            coarse_heap.push({dis, id});
            coarse_heap.pop();
        }
    };

    // ── 4× unrolled gather loop ──────────────────────────────────────────────
    // Each iteration processes 4 base vectors, issuing 4×8=32 gather loads.
    // Independent across vectors → OOO engine overlaps load latencies.
    size_t i = 0;
    for (; i + 4 <= N; i += 4) 
    {
        // Load 4 × 8 code bytes (4 consecutive vld1_u8)
        uint8x8_t cv0 = vld1_u8(codes + (i+0)*M);
        uint8x8_t cv1 = vld1_u8(codes + (i+1)*M);
        uint8x8_t cv2 = vld1_u8(codes + (i+2)*M);
        uint8x8_t cv3 = vld1_u8(codes + (i+3)*M);

        // Gather M=8 floats per vector into lo (subspaces 0-3) + hi (4-7)
        float32x4_t lo0 = neon_gather4(lut[0]+vget_lane_u8(cv0,0), lut[1]+vget_lane_u8(cv0,1),
                                        lut[2]+vget_lane_u8(cv0,2), lut[3]+vget_lane_u8(cv0,3));
        float32x4_t hi0 = neon_gather4(lut[4]+vget_lane_u8(cv0,4), lut[5]+vget_lane_u8(cv0,5),
                                        lut[6]+vget_lane_u8(cv0,6), lut[7]+vget_lane_u8(cv0,7));

        float32x4_t lo1 = neon_gather4(lut[0]+vget_lane_u8(cv1,0), lut[1]+vget_lane_u8(cv1,1),
                                        lut[2]+vget_lane_u8(cv1,2), lut[3]+vget_lane_u8(cv1,3));
        float32x4_t hi1 = neon_gather4(lut[4]+vget_lane_u8(cv1,4), lut[5]+vget_lane_u8(cv1,5),
                                        lut[6]+vget_lane_u8(cv1,6), lut[7]+vget_lane_u8(cv1,7));

        float32x4_t lo2 = neon_gather4(lut[0]+vget_lane_u8(cv2,0), lut[1]+vget_lane_u8(cv2,1),
                                        lut[2]+vget_lane_u8(cv2,2), lut[3]+vget_lane_u8(cv2,3));
        float32x4_t hi2 = neon_gather4(lut[4]+vget_lane_u8(cv2,4), lut[5]+vget_lane_u8(cv2,5),
                                        lut[6]+vget_lane_u8(cv2,6), lut[7]+vget_lane_u8(cv2,7));

        float32x4_t lo3 = neon_gather4(lut[0]+vget_lane_u8(cv3,0), lut[1]+vget_lane_u8(cv3,1),
                                        lut[2]+vget_lane_u8(cv3,2), lut[3]+vget_lane_u8(cv3,3));
        float32x4_t hi3 = neon_gather4(lut[4]+vget_lane_u8(cv3,4), lut[5]+vget_lane_u8(cv3,5),
                                        lut[6]+vget_lane_u8(cv3,6), lut[7]+vget_lane_u8(cv3,7));

        // Sum 8 gathered values → 1 approx_ip per vector
        float ip0 = vaddvq_f32(vaddq_f32(lo0, hi0));
        float ip1 = vaddvq_f32(vaddq_f32(lo1, hi1));
        float ip2 = vaddvq_f32(vaddq_f32(lo2, hi2));
        float ip3 = vaddvq_f32(vaddq_f32(lo3, hi3));

        heap_push(ip0, (uint32_t)(i+0));
        heap_push(ip1, (uint32_t)(i+1));
        heap_push(ip2, (uint32_t)(i+2));
        heap_push(ip3, (uint32_t)(i+3));
    }

    // ── Scalar tail for N % 4 remaining vectors ──────────────────────────────
    for (; i < N; ++i) 
    {
        uint8x8_t cv = vld1_u8(codes + i*M);
        float32x4_t lo = neon_gather4(lut[0]+vget_lane_u8(cv,0), lut[1]+vget_lane_u8(cv,1),
                                       lut[2]+vget_lane_u8(cv,2), lut[3]+vget_lane_u8(cv,3));
        float32x4_t hi = neon_gather4(lut[4]+vget_lane_u8(cv,4), lut[5]+vget_lane_u8(cv,5),
                                       lut[6]+vget_lane_u8(cv,6), lut[7]+vget_lane_u8(cv,7));
        heap_push(vaddvq_f32(vaddq_f32(lo, hi)), (uint32_t)i);
    }

    // ── Rerank (identical to pq_rerank_from_dtable) ──────────────────────────
    std::vector<uint32_t> cand_ids;
    cand_ids.reserve(p);
    while (!coarse_heap.empty()) 
    {
        cand_ids.push_back(coarse_heap.top().second);
        coarse_heap.pop();
    }

    std::priority_queue<std::pair<float, uint32_t>> result;
    for (uint32_t id : cand_ids) {
        const float* bv = base + (size_t)id * d;
        float ip = 0.0f;
        for (size_t j = 0; j < d; ++j) ip += bv[j] * query[j];
        float dis = 1.0f - ip;
        if (result.size() < k) {
            result.push({dis, id});
        } else if (dis < result.top().first) {
            result.push({dis, id});
            result.pop();
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// pq_flat_search_rerank_gather — two-phase PQ search: cc_unroll LUT + gather scan
//
// Pairs the best LUT build (cc_unroll, 0 reductions) with the gather coarse
// scan (eliminates the M=8 inner loop).  Uses PQIndexSIMD for the transposed
// centroid layout required by cc_unroll.
// ---------------------------------------------------------------------------
inline std::priority_queue<std::pair<float, uint32_t>>
pq_flat_search_rerank_gather(const PQIndexSIMD& pq_simd, const float* base,
                              const float* query, size_t k, size_t p)
{
    const PQIndex& index = *pq_simd.idx;
    std::vector<float> dtable(index.M * index.K);
    pq_build_lut_cc_unroll(pq_simd, query, dtable.data());
    return pq_rerank_from_dtable_gather(index, base, query, k, p, dtable.data());
}

// =============================================================================
// True matrix-matrix cache-blocked k-means (double-tiled: i_blk × k_blk)
//
// The previous pq_kmeans_simd only tiles k (for k_blk inside for i), which means
// the centroid tile is reloaded for EVERY vector — no reuse across vectors.
//
// True blocking interchanges the loops:
//   for i_blk:            ← batch BLOCK_N vectors
//     for k_blk:          ← centroid tile (3 KB) loaded ONCE here
//       for i in i_blk:   ← reused BLOCK_N times before next k_blk
//
// Centroid tile: BLOCK_K × dsub × 4 = 64 × 12 × 4 = 3 KB  (stays in L1)
// Data tile:     BLOCK_N × dsub × 4 = 128 × 12 × 4 = 6 KB  (accessed per i_blk)
// IP buffer:     BLOCK_N × K   × 4 = 128 × 256× 4 = 128 KB (written once per i_blk)
//
// The centroid tile is amortised across BLOCK_N=128 vectors, reducing centroid
// traffic from n loads to ceil(n/BLOCK_N) loads per k-block.
//
// Preconditions: K % 16 == 0, BLOCK_K % 16 == 0  (both satisfied for K=256)
// =============================================================================

// pq_kmeans_simd_blocked — true double-tiled SIMD k-means (i_blk × k_blk)
static inline void pq_kmeans_simd_blocked(const float* data, size_t n, size_t dsub,
                                           size_t K, int n_iters, float* centroids)
{
    // LCG-shuffle initialization (identical to pq_kmeans_simd)
    std::vector<size_t> perm(n);
    for (size_t i = 0; i < n; ++i) perm[i] = i;
    unsigned rng = 0xA5A5A5A5u;
    for (size_t i = n - 1; i > 0; --i)
    {
        rng = rng * 1664525u + 1013904223u;
        size_t j = rng % (i + 1);
        size_t tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
    }
    for (size_t k = 0; k < K; ++k)
        memcpy(centroids + k * dsub, data + perm[k] * dsub, dsub * sizeof(float));

    std::vector<uint32_t> assign(n);
    std::vector<float>    cnt(K);
    std::vector<float>    cents_t(dsub * K);  // transposed layout [j][k]
    std::vector<float>    norm2_c(K);
    // Persistent i-block buffers — allocated once, reused each i_blk
    std::vector<float>    ip_buf(PQ_KMEANS_BLOCK_N * K);
    std::vector<float>    norm2_v_buf(PQ_KMEANS_BLOCK_N);

    for (int iter = 0; iter < n_iters; ++iter)
    {
        // Transpose centroids [k][j] → [j][k] and precompute ||c_k||²
        for (size_t k = 0; k < K; ++k)
        {
            float n2 = 0.0f;
            for (size_t j = 0; j < dsub; ++j)
            {
                float v = centroids[k * dsub + j];
                cents_t[j * K + k] = v;
                n2 += v * v;
            }
            norm2_c[k] = n2;
        }

        // Assign step: TRUE double-tiled blocking (i_blk outer, k_blk inner)
        for (size_t i_blk = 0; i_blk < n; i_blk += PQ_KMEANS_BLOCK_N)
        {
            const size_t i_end = std::min(i_blk + PQ_KMEANS_BLOCK_N, n);
            const size_t ni    = i_end - i_blk;

            // Precompute ||v||² for all vectors in this i-block
            for (size_t ii = 0; ii < ni; ++ii)
            {
                const float* v = data + (i_blk + ii) * dsub;
                float n2 = 0.0f;
                for (size_t j = 0; j < dsub; ++j) n2 += v[j] * v[j];
                norm2_v_buf[ii] = n2;
            }

            // For each centroid block, sweep all ni vectors against it.
            // The centroid tile (~3 KB) is loaded into L1 here and stays hot
            // for ni=128 consecutive vector iterations before the next k_blk.
            for (size_t k_blk = 0; k_blk < K; k_blk += PQ_KMEANS_BLOCK_K)
            {
                for (size_t ii = 0; ii < ni; ++ii)
                {
                    const float* v      = data + (i_blk + ii) * dsub;
                    float*       ip_row = ip_buf.data() + ii * K;

                    // 4× unrolled cross-centroid SIMD: 16 centroids per iteration
                    for (size_t k = k_blk; k < k_blk + PQ_KMEANS_BLOCK_K; k += 16)
                    {
                        float32x4_t acc0 = vdupq_n_f32(0.0f);
                        float32x4_t acc1 = vdupq_n_f32(0.0f);
                        float32x4_t acc2 = vdupq_n_f32(0.0f);
                        float32x4_t acc3 = vdupq_n_f32(0.0f);
                        for (size_t j = 0; j < dsub; ++j)
                        {
                            const float        v_j = v[j];
                            const float* const ptr = cents_t.data() + j * K + k;
                            acc0 = vmlaq_n_f32(acc0, vld1q_f32(ptr),      v_j);
                            acc1 = vmlaq_n_f32(acc1, vld1q_f32(ptr +  4), v_j);
                            acc2 = vmlaq_n_f32(acc2, vld1q_f32(ptr +  8), v_j);
                            acc3 = vmlaq_n_f32(acc3, vld1q_f32(ptr + 12), v_j);
                        }
                        vst1q_f32(ip_row + k,      acc0);
                        vst1q_f32(ip_row + k +  4, acc1);
                        vst1q_f32(ip_row + k +  8, acc2);
                        vst1q_f32(ip_row + k + 12, acc3);
                    }
                }
            }

            // Argmin: find nearest centroid for each vector in the i-block
            for (size_t ii = 0; ii < ni; ++ii)
            {
                const float* ip_row  = ip_buf.data() + ii * K;
                float        best_d2 = std::numeric_limits<float>::max();
                uint32_t     best_k  = 0;
                for (size_t k = 0; k < K; ++k)
                {
                    float d2 = norm2_v_buf[ii] + norm2_c[k] - 2.0f * ip_row[k];
                    if (d2 < best_d2) { best_d2 = d2; best_k = static_cast<uint32_t>(k); }
                }
                assign[i_blk + ii] = best_k;
            }
        }

        // Update step: recompute centroids as cluster means (scalar)
        memset(centroids, 0, K * dsub * sizeof(float));
        std::fill(cnt.begin(), cnt.end(), 0.0f);
        for (size_t i = 0; i < n; ++i)
        {
            const float* v = data + i * dsub;
            float*       c = centroids + assign[i] * dsub;
            for (size_t j = 0; j < dsub; ++j) c[j] += v[j];
            cnt[assign[i]] += 1.0f;
        }
        for (size_t k = 0; k < K; ++k)
        {
            if (cnt[k] > 0.0f)
            {
                float inv = 1.0f / cnt[k];
                float* c  = centroids + k * dsub;
                for (size_t j = 0; j < dsub; ++j) c[j] *= inv;
            }
        }
    }
}

// pq_build_index_simd_blocked — builds a PQIndex using pq_kmeans_simd_blocked
// Uses true double-tiled (i_blk × k_blk) blocking; original functions untouched.
inline void pq_build_index_simd_blocked(PQIndex& out, const float* base, size_t n, size_t d,
                                         size_t m_subs = 8, size_t k_cents = 256, int n_iters = 25)
{
    out.M           = m_subs;
    out.K           = k_cents;
    out.dsub        = d / m_subs;
    out.vecdim      = d;
    out.base_number = n;

    out.centroids.resize(m_subs * k_cents * out.dsub, 0.0f);
    out.codes.resize(n * m_subs);

    std::vector<float> sub_data(n * out.dsub);

    for (size_t m = 0; m < m_subs; ++m) 
    {
        for (size_t i = 0; i < n; ++i)
            memcpy(sub_data.data() + i * out.dsub,
                   base + i * d + m * out.dsub,
                   out.dsub * sizeof(float));

        float* cents_m = out.centroids.data() + m * k_cents * out.dsub;
        pq_kmeans_simd_blocked(sub_data.data(), n, out.dsub, k_cents, n_iters, cents_m);

        // Encode: assign each base vector to its nearest centroid
        for (size_t i = 0; i < n; ++i) 
        {
            const float* v    = sub_data.data() + i * out.dsub;
            float        best = std::numeric_limits<float>::max();
            uint8_t      bk   = 0;
            for (size_t k = 0; k < k_cents; ++k) {
                const float* c = cents_m + k * out.dsub;
                float d2 = 0.0f;
                for (size_t j = 0; j < out.dsub; ++j) {
                    float diff = v[j] - c[j];
                    d2 += diff * diff;
                }
                if (d2 < best) { best = d2; bk = static_cast<uint8_t>(k); }
            }
            out.codes[i * m_subs + m] = bk;
        }
    }
}
