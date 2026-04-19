// =============================================================================
// flat_simd.h — NEON-vectorized flat k-NN scan (ARM, SIMD data-parallelism)
//
// "Parallel" here means SIMD data-level parallelism: one instruction operates
// on 8 float32 lanes simultaneously, not multiple OS threads.
//
// Speedup over flat_scan_normal.h:
//   - Main loop: 8× throughput via 8-wide NEON (float32x4x2_t)
//   - vmlaq_f32 fuses multiply+accumulate into a single instruction
//   - vaddvq_f32 (AArch64) reduces 4 lanes to scalar in one instruction
//   - dim=96 is divisible by 8 → 12 clean iterations, no scalar tail needed
//
// Platform: AArch64. NEON is mandatory on AArch64; no extra -march flag needed.
// Compile:  g++ main.cc -o main -O2 -fopenmp -lpthread -std=c++11
// =============================================================================
#pragma once
#include <arm_neon.h>
#include <queue>
#include <utility>
#include <cstdint>

// simd8float32 — 8 × float32 SIMD abstraction built on NEON
//
// NEON registers are 128 bits wide, holding 4 × float32 each.
// float32x4x2_t bundles two consecutive registers to expose 8 float lanes,
// matching the 8-stride loops in the inner product kernel below.
struct simd8float32
{
    float32x4x2_t data;  // data.val[0]: lanes 0-3 | data.val[1]: lanes 4-7

    simd8float32() = default;

    // Broadcast a single scalar to all 8 lanes
    // Used to zero-initialise the accumulator: simd8float32 sum(0.0f)
    explicit simd8float32(float val) {
        // vdupq_n_f32: copy one scalar into all 4 lanes of a 128-bit register
        data.val[0] = vdupq_n_f32(val);
        data.val[1] = vdupq_n_f32(val);
    }

    // Load 8 consecutive floats from memory
    // Unaligned access is permitted and penalty-free on AArch64
    explicit simd8float32(const float* x) {
        // vld1q_f32: load 4 floats from ptr into one 128-bit NEON register
        data.val[0] = vld1q_f32(x);      // floats [0..3]
        data.val[1] = vld1q_f32(x + 4);  // floats [4..7]
    }

    // Elementwise multiply: each of the 8 lanes multiplied independently
    simd8float32 operator*(const simd8float32& o) const {
        simd8float32 r;
        // vmulq_f32: 4-wide float multiply — one NEON instruction per register pair
        r.data.val[0] = vmulq_f32(data.val[0], o.data.val[0]);
        r.data.val[1] = vmulq_f32(data.val[1], o.data.val[1]);
        return r;
    }

    // Elementwise add (used in operator+= form)
    simd8float32& operator+=(const simd8float32& o) {
        data.val[0] = vaddq_f32(data.val[0], o.data.val[0]);
        data.val[1] = vaddq_f32(data.val[1], o.data.val[1]);
        return *this;
    }

    // Fused multiply-accumulate: *this += a * b
    //
    // vmlaq_f32(acc, x, y) = acc + x*y in ONE instruction.
    // On in-order ARM pipelines this saves one cycle vs. separate vmulq + vaddq
    // because no intermediate register is needed to hold the product.
    simd8float32& madd(const simd8float32& a, const simd8float32& b) {
        data.val[0] = vmlaq_f32(data.val[0], a.data.val[0], b.data.val[0]);
        data.val[1] = vmlaq_f32(data.val[1], a.data.val[1], b.data.val[1]);
        return *this;
    }

    // Store 8 floats to memory (used in horizontal reduce)
    void storeu(float* dst) const {
        vst1q_f32(dst,     data.val[0]);  // lanes 0-3
        vst1q_f32(dst + 4, data.val[1]);  // lanes 4-7
    }
};

// simd_inner_product_neon — dot product of two dim-length float32 vectors
//
// Main loop: 8 floats/iteration via simd8float32 + vmlaq_f32.
// Horizontal reduce: vaddvq_f32 sums all 4 lanes of a 128-bit register in
// one instruction (AArch64 only; faster than chaining vpadd_f32).
//
// Precondition: dim % 8 == 0  (DEEP100K: dim=96, 96÷8 = 12 iterations)
inline float simd_inner_product_neon(const float* a, const float* b, size_t dim)
{
    simd8float32 sum(0.0f);  // 8-lane accumulator, all lanes initialised to 0.0

    for (size_t i = 0; i < dim; i += 8) {
        simd8float32 va(a + i);  // load 8 floats from vector a
        simd8float32 vb(b + i);  // load 8 floats from vector b
        sum.madd(va, vb);        // sum += va * vb  (vmlaq_f32 × 2 registers)
    }

    // Horizontal reduce: collapse 8 lanes → scalar.
    // vaddvq_f32: sums all 4 lanes of a 128-bit register into one float.
    float s0 = vaddvq_f32(sum.data.val[0]);  // sum of lanes 0-3
    float s1 = vaddvq_f32(sum.data.val[1]);  // sum of lanes 4-7
    return s0 + s1;
}

// simd_flat_search — exhaustive k-NN using NEON-vectorized inner product
//
// Drop-in replacement for flat_search_normal() / flat_search().
// Max-heap maintenance logic is identical; only the distance kernel changes.
// recall@k = 1.0 because this remains an exact brute-force scan.
std::priority_queue<std::pair<float, uint32_t>>
simd_flat_search(float* base, float* query,
                 size_t base_number, size_t vecdim, size_t k)
{
    // Max-heap of (distance, index): top = current farthest among top-k
    std::priority_queue<std::pair<float, uint32_t>> q;

    for (size_t i = 0; i < base_number; ++i) {
        // IP distance = 1 - dot(base_i, query), computed 8 floats at a time
        float dis = 1.0f - simd_inner_product_neon(base + i * vecdim, query, vecdim);

        if (q.size() < k) {
            q.push({dis, (uint32_t)i});
        } else if (dis < q.top().first) {
            // Replace the farthest candidate with this closer vector
            q.push({dis, (uint32_t)i});
            q.pop();
        }
    }
    return q;
}
