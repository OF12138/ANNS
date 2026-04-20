// =============================================================================
// sq_multibit_simd.h — SQ-SIMD at multiple quantization bit-widths (4/8/16)
//
// Used by exp/SQtradeoff.cc to sweep the recall-latency tradeoff vs. both
//   • p    : coarse-scan candidate count (top-p → rerank → top-k)
//   • bits : 4, 8, or 16  (codes per dimension)
//
// All three variants share the same approximate-dot formula:
//   dot(base, query) ≈ offset + Σ code[j] * adj_query[j]
// only the bit-width of code[j] and its SIMD load/widen cascade differ.
//
// Storage (per base vector of dim d):
//     bits=4 :  ceil(d/2) bytes   (two codes packed per byte, low nibble first)
//     bits=8 :  d bytes
//     bits=16:  2*d bytes
//
// Preconditions for the SIMD kernels:
//     bits=4 : d % 32 == 0   (96 / 32 = 3 iterations for DEEP100K)
//     bits=8 : d % 16 == 0   (96 / 16 = 6 iterations)
//     bits=16: d % 16 == 0   (96 / 16 = 6 iterations)
//
// The 8-bit SIMD search reuses sq_flat_search_simd() from
// ARM/Alg_parallel/sq_flat_simd.h — no re-implementation here.
// =============================================================================
#pragma once
#include <arm_neon.h>
#include <queue>
#include <utility>
#include <cstdint>
#include <vector>
#include <limits>
#include "../ARM/Alg_parallel/sq_flat_simd.h"   // SQIndex + sq_flat_search_simd (8-bit)

// ---------------------------------------------------------------------------
// 4-bit index: two codes packed per byte
// ---------------------------------------------------------------------------
struct SQIndex4 {
    std::vector<uint8_t> codes;    // size = base_number * (vecdim/2)
    std::vector<float>   mins;
    std::vector<float>   scales;
    size_t base_number = 0;
    size_t vecdim      = 0;
    static constexpr int N_MAX = 15;  // (1 << 4) - 1

    void build(const float* base, size_t n, size_t d) {
        base_number = n;
        vecdim      = d;
        mins.resize(d);
        scales.resize(d);
        codes.assign(n * (d / 2), 0);   // assumes d is even

        for (size_t j = 0; j < d; ++j) {
            float mn =  std::numeric_limits<float>::max();
            float mx = -std::numeric_limits<float>::max();
            for (size_t i = 0; i < n; ++i) {
                float v = base[i * d + j];
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            mins[j]   = mn;
            scales[j] = (mx > mn) ? (mx - mn) / float(N_MAX) : 1.0f;
        }

        for (size_t i = 0; i < n; ++i) {
            uint8_t* dst = codes.data() + i * (d / 2);
            for (size_t j = 0; j < d; j += 2) {
                int q0 = int((base[i*d + j    ] - mins[j    ]) / scales[j    ] + 0.5f);
                int q1 = int((base[i*d + j + 1] - mins[j + 1]) / scales[j + 1] + 0.5f);
                if (q0 < 0) q0 = 0; if (q0 > N_MAX) q0 = N_MAX;
                if (q1 < 0) q1 = 0; if (q1 > N_MAX) q1 = N_MAX;
                // low nibble = code for even dim j, high nibble = code for odd dim j+1
                dst[j / 2] = static_cast<uint8_t>((q1 << 4) | q0);
            }
        }
    }

    size_t index_bytes() const { return codes.size(); }
};

// ---------------------------------------------------------------------------
// 16-bit index: one uint16 per code
// ---------------------------------------------------------------------------
struct SQIndex16 {
    std::vector<uint16_t> codes;   // size = base_number * vecdim
    std::vector<float>    mins;
    std::vector<float>    scales;
    size_t base_number = 0;
    size_t vecdim      = 0;
    static constexpr int N_MAX = 65535;  // (1 << 16) - 1

    void build(const float* base, size_t n, size_t d) {
        base_number = n;
        vecdim      = d;
        mins.resize(d);
        scales.resize(d);
        codes.resize(n * d);

        for (size_t j = 0; j < d; ++j) {
            float mn =  std::numeric_limits<float>::max();
            float mx = -std::numeric_limits<float>::max();
            for (size_t i = 0; i < n; ++i) {
                float v = base[i * d + j];
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            mins[j]   = mn;
            scales[j] = (mx > mn) ? (mx - mn) / float(N_MAX) : 1.0f;
        }

        for (size_t i = 0; i < n; ++i) {
            uint16_t* dst = codes.data() + i * d;
            for (size_t j = 0; j < d; ++j) {
                int q = int((base[i*d + j] - mins[j]) / scales[j] + 0.5f);
                if (q < 0)     q = 0;
                if (q > N_MAX) q = N_MAX;
                dst[j] = static_cast<uint16_t>(q);
            }
        }
    }

    size_t index_bytes() const { return codes.size() * sizeof(uint16_t); }
};

// ---------------------------------------------------------------------------
// 4-bit SIMD coarse dot
//   Load 16 packed bytes → 32 codes (unpack nibbles via AND + SHR).
//   vzip1q_u8/vzip2q_u8 interleaves low/high nibbles back into sequential order:
//     result[0]=lo[0], result[1]=hi[0], result[2]=lo[1], ...
//   Dim=96: processes 32 codes/iter → 3 iterations with 8 FMAs each.
// ---------------------------------------------------------------------------
inline float sq4_coarse_dot_simd(const uint8_t* packed, const float* adj_query, size_t dim)
{
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);

    const uint8x16_t mask_lo = vdupq_n_u8(0x0F);

    for (size_t j = 0; j < dim; j += 32)
    {
        // 16 packed bytes = 32 codes
        uint8x16_t bytes = vld1q_u8(packed + j / 2);
        uint8x16_t lo    = vandq_u8(bytes, mask_lo);   // codes at even dims
        uint8x16_t hi    = vshrq_n_u8(bytes, 4);       // codes at odd dims

        // Interleave back to sequential order (codes 0..15 and 16..31)
        uint8x16_t seq_a = vzip1q_u8(lo, hi);   // codes j..j+15
        uint8x16_t seq_b = vzip2q_u8(lo, hi);   // codes j+16..j+31

        // ----- process codes j..j+15 -----
        uint16x8_t wA_lo = vmovl_u8(vget_low_u8 (seq_a));
        uint16x8_t wA_hi = vmovl_u8(vget_high_u8(seq_a));
        sum0 = vmlaq_f32(sum0,
                         vcvtq_f32_u32(vmovl_u16(vget_low_u16 (wA_lo))),
                         vld1q_f32(adj_query + j));
        sum1 = vmlaq_f32(sum1,
                         vcvtq_f32_u32(vmovl_u16(vget_high_u16(wA_lo))),
                         vld1q_f32(adj_query + j +  4));
        sum2 = vmlaq_f32(sum2,
                         vcvtq_f32_u32(vmovl_u16(vget_low_u16 (wA_hi))),
                         vld1q_f32(adj_query + j +  8));
        sum3 = vmlaq_f32(sum3,
                         vcvtq_f32_u32(vmovl_u16(vget_high_u16(wA_hi))),
                         vld1q_f32(adj_query + j + 12));

        // ----- process codes j+16..j+31 -----
        uint16x8_t wB_lo = vmovl_u8(vget_low_u8 (seq_b));
        uint16x8_t wB_hi = vmovl_u8(vget_high_u8(seq_b));
        sum0 = vmlaq_f32(sum0,
                         vcvtq_f32_u32(vmovl_u16(vget_low_u16 (wB_lo))),
                         vld1q_f32(adj_query + j + 16));
        sum1 = vmlaq_f32(sum1,
                         vcvtq_f32_u32(vmovl_u16(vget_high_u16(wB_lo))),
                         vld1q_f32(adj_query + j + 20));
        sum2 = vmlaq_f32(sum2,
                         vcvtq_f32_u32(vmovl_u16(vget_low_u16 (wB_hi))),
                         vld1q_f32(adj_query + j + 24));
        sum3 = vmlaq_f32(sum3,
                         vcvtq_f32_u32(vmovl_u16(vget_high_u16(wB_hi))),
                         vld1q_f32(adj_query + j + 28));
    }

    sum0 = vaddq_f32(sum0, sum1);
    sum2 = vaddq_f32(sum2, sum3);
    return vaddvq_f32(vaddq_f32(sum0, sum2));
}

// ---------------------------------------------------------------------------
// 16-bit SIMD coarse dot
//   Load 8 uint16 codes per Q register (uint16x8_t). Widen uint16 → uint32 → float32.
//   Dim=96: processes 16 codes/iter → 6 iterations with 4 FMAs each.
// ---------------------------------------------------------------------------
inline float sq16_coarse_dot_simd(const uint16_t* code, const float* adj_query, size_t dim)
{
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);

    for (size_t j = 0; j < dim; j += 16)
    {
        uint16x8_t c_a = vld1q_u16(code + j);       // codes 0..7
        uint16x8_t c_b = vld1q_u16(code + j + 8);   // codes 8..15

        float32x4_t f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16 (c_a)));
        float32x4_t f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(c_a)));
        float32x4_t f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16 (c_b)));
        float32x4_t f3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(c_b)));

        sum0 = vmlaq_f32(sum0, f0, vld1q_f32(adj_query + j));
        sum1 = vmlaq_f32(sum1, f1, vld1q_f32(adj_query + j +  4));
        sum2 = vmlaq_f32(sum2, f2, vld1q_f32(adj_query + j +  8));
        sum3 = vmlaq_f32(sum3, f3, vld1q_f32(adj_query + j + 12));
    }

    sum0 = vaddq_f32(sum0, sum1);
    sum2 = vaddq_f32(sum2, sum3);
    return vaddvq_f32(vaddq_f32(sum0, sum2));
}

// ---------------------------------------------------------------------------
// 4-bit two-phase search
// ---------------------------------------------------------------------------
inline std::priority_queue<std::pair<float, uint32_t>>
sq_search_4bit_simd(const SQIndex4& index, const float* base,
                    const float* query, size_t k, size_t p)
{
    const size_t d = index.vecdim;

    std::vector<float> adj_query(d);
    float offset = 0.0f;
    for (size_t j = 0; j < d; ++j) {
        adj_query[j] = index.scales[j] * query[j];
        offset      += index.mins[j]   * query[j];
    }

    std::priority_queue<std::pair<float, uint32_t>> coarse;
    const size_t stride = d / 2;   // bytes per packed vector
    for (size_t i = 0; i < index.base_number; ++i) {
        float dot = offset + sq4_coarse_dot_simd(
            index.codes.data() + i * stride, adj_query.data(), d);
        float dis = 1.0f - dot;
        if (coarse.size() < p) {
            coarse.push({dis, static_cast<uint32_t>(i)});
        } else if (dis < coarse.top().first) {
            coarse.push({dis, static_cast<uint32_t>(i)});
            coarse.pop();
        }
    }

    std::vector<uint32_t> ids;
    ids.reserve(p);
    while (!coarse.empty()) { ids.push_back(coarse.top().second); coarse.pop(); }

    std::priority_queue<std::pair<float, uint32_t>> result;
    for (uint32_t id : ids) {
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

// ---------------------------------------------------------------------------
// 16-bit two-phase search
// ---------------------------------------------------------------------------
inline std::priority_queue<std::pair<float, uint32_t>>
sq_search_16bit_simd(const SQIndex16& index, const float* base,
                     const float* query, size_t k, size_t p)
{
    const size_t d = index.vecdim;

    std::vector<float> adj_query(d);
    float offset = 0.0f;
    for (size_t j = 0; j < d; ++j) {
        adj_query[j] = index.scales[j] * query[j];
        offset      += index.mins[j]   * query[j];
    }

    std::priority_queue<std::pair<float, uint32_t>> coarse;
    for (size_t i = 0; i < index.base_number; ++i) {
        float dot = offset + sq16_coarse_dot_simd(
            index.codes.data() + i * d, adj_query.data(), d);
        float dis = 1.0f - dot;
        if (coarse.size() < p) {
            coarse.push({dis, static_cast<uint32_t>(i)});
        } else if (dis < coarse.top().first) {
            coarse.push({dis, static_cast<uint32_t>(i)});
            coarse.pop();
        }
    }

    std::vector<uint32_t> ids;
    ids.reserve(p);
    while (!coarse.empty()) { ids.push_back(coarse.top().second); coarse.pop(); }

    std::priority_queue<std::pair<float, uint32_t>> result;
    for (uint32_t id : ids) {
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
