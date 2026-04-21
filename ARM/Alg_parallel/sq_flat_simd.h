// =============================================================================
// sq_flat_simd.h — NEON-vectorized Scalar Quantization flat k-NN scan (8-bit)
//
// SIMD counterpart of ARM/Alg_normal/sq_flat_normal.h. Reuses the same SQIndex
// (one uint8 code per dimension). Only the scan kernels change.
//
// Approximate IP dot (identical to the scalar version):
//   dot(base_i, query) ≈ offset + Σ code[j] * adj_query[j]
//   offset       = Σ min[j]   * query[j]   (precomputed per query)
//   adj_query[j] = scale[j]  * query[j]    (precomputed per query)
//
// SIMD strategy for the Σ term (coarse scan):
//   1. vld1q_u8 : load 16 uint8 codes into one 128-bit Q register
//   2. widen uint8 → uint16 via vmovl_u8 (split into low-half / high-half)
//   3. widen uint16 → uint32 via vmovl_u16
//   4. vcvtq_f32_u32 : convert uint32 → float32
//   5. vmlaq_f32 : FMA against 4 × adj_query float32x4_t loads
//   4 independent accumulators hide the ~4-cycle FMA latency (same trick as
//   simd_flat_search_unroll in flat_simd.h).
//
// Rerank is plain float32 IP with the same 4-accumulator unroll.
// Precondition: vecdim % 16 == 0  (DEEP100K dim=96 → 6 clean iterations).
// =============================================================================
#pragma once
#include <arm_neon.h>
#include <queue>
#include <utility>
#include <cstdint>
#include <vector>
#include "../Alg_normal/sq_flat_normal.h"   // pulls in SQIndex definition

// sq8_coarse_dot_simd — partial sum Σ code[j] * adj_query[j] using NEON
// Returns the SIMD-accumulated partial sum (caller adds offset).
inline float sq8_coarse_dot_simd(const uint8_t* code, const float* adj_query, size_t dim)
{
    // 4 independent float accumulators (no dependency → full FMA throughput)
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);

    for (size_t j = 0; j < dim; j += 16)
    {
        // Load 16 uint8 codes in a single 128-bit register
        uint8x16_t c8 = vld1q_u8(code + j);

        // Widen 16 × uint8  →  two 8 × uint16 vectors (low half, high half)
        uint16x8_t c16_lo = vmovl_u8(vget_low_u8(c8));    // codes 0..7
        uint16x8_t c16_hi = vmovl_u8(vget_high_u8(c8));   // codes 8..15

        // Widen 8 × uint16 →  four 4 × uint32 → four 4 × float32
        float32x4_t f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16 (c16_lo)));  // 0..3
        float32x4_t f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(c16_lo)));  // 4..7
        float32x4_t f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16 (c16_hi)));  // 8..11
        float32x4_t f3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(c16_hi)));  // 12..15

        // FMA against the matching adj_query slice (16 floats total)
        sum0 = vmlaq_f32(sum0, f0, vld1q_f32(adj_query + j));
        sum1 = vmlaq_f32(sum1, f1, vld1q_f32(adj_query + j +  4));
        sum2 = vmlaq_f32(sum2, f2, vld1q_f32(adj_query + j +  8));
        sum3 = vmlaq_f32(sum3, f3, vld1q_f32(adj_query + j + 12));
    }

    sum0 = vaddq_f32(sum0, sum1);
    sum2 = vaddq_f32(sum2, sum3);
    return vaddvq_f32(vaddq_f32(sum0, sum2));   // horizontal reduce 4 lanes → scalar
}

// sq_rerank_dot_simd — exact float32 IP with 4-unroll FMA (for Phase 2 rerank)
inline float sq_rerank_dot_simd(const float* a, const float* b, size_t dim)
{
    float32x4_t s0 = vdupq_n_f32(0.0f);
    float32x4_t s1 = vdupq_n_f32(0.0f);
    float32x4_t s2 = vdupq_n_f32(0.0f);
    float32x4_t s3 = vdupq_n_f32(0.0f);
    for (size_t j = 0; j < dim; j += 16) 
    {
        s0 = vmlaq_f32(s0, vld1q_f32(a + j),      vld1q_f32(b + j));
        s1 = vmlaq_f32(s1, vld1q_f32(a + j +  4), vld1q_f32(b + j +  4));
        s2 = vmlaq_f32(s2, vld1q_f32(a + j +  8), vld1q_f32(b + j +  8));
        s3 = vmlaq_f32(s3, vld1q_f32(a + j + 12), vld1q_f32(b + j + 12));
    }
    s0 = vaddq_f32(s0, s1);
    s2 = vaddq_f32(s2, s3);
    return vaddvq_f32(vaddq_f32(s0, s2));
}

// sq_flat_search_simd — two-phase SQ k-NN with NEON coarse scan + SIMD rerank
//
// Drop-in replacement for sq_flat_search_normal(); identical interface.
//   index : pre-built SQIndex (8-bit codes)
//   base  : original float32 base vectors (for exact rerank)
//   query : float32 query vector
//   k     : number of nearest neighbours to return
//   p     : coarse-scan candidate count (p ≥ k)
std::priority_queue<std::pair<float, uint32_t>>
sq_flat_search_simd(const SQIndex& index, const float* base,
                    const float* query, size_t k, size_t p)
{
    const size_t d = index.vecdim;

    // Precompute offset and adj_query (identical to the scalar SQ version)
    std::vector<float> adj_query(d);
    float offset = 0.0f;
    for (size_t j = 0; j < d; ++j) {
        adj_query[j] = index.scales[j] * query[j];
        offset      += index.mins[j]   * query[j];
    }

    // ---- Phase 1: coarse scan over uint8 codes, keep top-p --------------------
    std::priority_queue<std::pair<float, uint32_t>> coarse_heap;
    for (size_t i = 0; i < index.base_number; ++i) {
        float dot = offset + sq8_coarse_dot_simd(
            index.codes.data() + i * d, adj_query.data(), d);
        float dis = 1.0f - dot;

        if (coarse_heap.size() < p) {
            coarse_heap.push({dis, static_cast<uint32_t>(i)});
        } else if (dis < coarse_heap.top().first) {
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

    // ---- Phase 2: exact rerank on float32 base vectors -----------------------
    std::priority_queue<std::pair<float, uint32_t>> result;
    for (uint32_t id : cand_ids) {
        float dot = sq_rerank_dot_simd(base + static_cast<size_t>(id) * d, query, d);
        float dis = 1.0f - dot;

        if (result.size() < k) {
            result.push({dis, id});
        } else if (dis < result.top().first) {
            result.push({dis, id});
            result.pop();
        }
    }
    return result;
}

// =============================================================================
// SDC (Symmetric Distance Computation) variant
//
// The query is quantized to uint8 using the same per-dimension min/scale as
// the base index.  The coarse scan then computes an integer dot product:
//   idot = Σ base_code[j] * qcode[j]   (uint8 × uint8 → uint32)
// which approximates the relative IP ranking without any float multiply in
// the inner loop.
//
// NEON kernel: vmull_u8 → 8 uint16 products per call (two calls = 16 muls per
// 16-byte load).  uint16 results are widened to uint32 via vmovl_u16 before
// accumulation to avoid overflow (max lane after dim=96: 6×65025 = 390K < 4G).
// Four independent uint32x4 accumulators break the dependency chain.
//
// The integer dot is negated so the existing max-heap naturally retains the
// p candidates with the HIGHEST integer dot (= best approximate IP).
// Rerank phase is identical to sq_flat_search_simd.
// =============================================================================

// Quantize a float32 query into uint8 codes using the SQIndex per-dim min/scale.
inline void sq_quantize_query(const SQIndex& index, const float* query,
                               uint8_t* qcodes)
{
    const size_t d = index.vecdim;
    for (size_t j = 0; j < d; ++j) 
    {
        int q = (int)((query[j] - index.mins[j]) / index.scales[j] + 0.5f);
        if (q < 0)   q = 0;
        if (q > 255) q = 255;
        qcodes[j] = (uint8_t)q;
    }
}

// sdc_coarse_dot_simd — integer dot product uint8×uint8 → uint32 via NEON
// Processes 16 code pairs per iteration (vmull_u8 ×2, then vmovl_u16 ×4).
inline float sdc_coarse_dot_simd(const uint8_t* base_code, const uint8_t* qcode, size_t dim)
{
    uint32x4_t sum0 = vdupq_n_u32(0);
    uint32x4_t sum1 = vdupq_n_u32(0);
    uint32x4_t sum2 = vdupq_n_u32(0);
    uint32x4_t sum3 = vdupq_n_u32(0);

    for (size_t j = 0; j < dim; j += 16) 
    {
        uint8x16_t a = vld1q_u8(base_code + j);
        uint8x16_t b = vld1q_u8(qcode + j);

        uint16x8_t p_lo = vmull_u8(vget_low_u8(a),  vget_low_u8(b));   // codes 0..7
        uint16x8_t p_hi = vmull_u8(vget_high_u8(a), vget_high_u8(b));  // codes 8..15

        sum0 = vaddq_u32(sum0, vmovl_u16(vget_low_u16(p_lo)));
        sum1 = vaddq_u32(sum1, vmovl_u16(vget_high_u16(p_lo)));
        sum2 = vaddq_u32(sum2, vmovl_u16(vget_low_u16(p_hi)));
        sum3 = vaddq_u32(sum3, vmovl_u16(vget_high_u16(p_hi)));
    }

    sum0 = vaddq_u32(sum0, sum1);
    sum2 = vaddq_u32(sum2, sum3);
    return (float)vaddvq_u32(vaddq_u32(sum0, sum2));
}

// sq_flat_search_sdc — two-phase SQ k-NN with SDC integer coarse scan
inline std::priority_queue<std::pair<float, uint32_t>>
sq_flat_search_sdc(const SQIndex& index, const float* base,
                   const float* query, size_t k, size_t p)
{
    const size_t d = index.vecdim;

    std::vector<uint8_t> qcodes(d);
    sq_quantize_query(index, query, qcodes.data());

    // Phase 1: integer coarse scan — negate idot so max-heap keeps top-p by highest IP
    std::priority_queue<std::pair<float, uint32_t>> coarse_heap;
    for (size_t i = 0; i < index.base_number; ++i) 
    {
        float neg_idot = -sdc_coarse_dot_simd(
            index.codes.data() + i * d, qcodes.data(), d);

        if (coarse_heap.size() < p) {
            coarse_heap.push({neg_idot, static_cast<uint32_t>(i)});
        } else if (neg_idot < coarse_heap.top().first) {
            coarse_heap.push({neg_idot, static_cast<uint32_t>(i)});
            coarse_heap.pop();
        }
    }

    std::vector<uint32_t> cand_ids;
    cand_ids.reserve(p);
    while (!coarse_heap.empty()) {
        cand_ids.push_back(coarse_heap.top().second);
        coarse_heap.pop();
    }

    // Phase 2: exact float32 rerank (identical to sq_flat_search_simd)
    std::priority_queue<std::pair<float, uint32_t>> result;
    for (uint32_t id : cand_ids) 
    {
        float dot = sq_rerank_dot_simd(base + static_cast<size_t>(id) * d, query, d);
        float dis = 1.0f - dot;

        if (result.size() < k) {
            result.push({dis, id});
        } else if (dis < result.top().first) {
            result.push({dis, id});
            result.pop();
        }
    }
    return result;
}
