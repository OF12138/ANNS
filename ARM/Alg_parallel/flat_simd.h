// =============================================================================
// flat_simd.h — NEON-vectorized flat k-NN scan (ARM, SIMD data-parallelism)
//
// "Parallel" here means SIMD data-level parallelism: one instruction operates
// on 4 float32 lanes simultaneously using a single 128-bit Q register.
//
// Speedup over flat_scan_normal.h:
//   - Main loop: 4× throughput via one float32x4_t Q register per iteration
//   - vmlaq_f32 fuses multiply+accumulate into a single instruction
//   - vaddvq_f32 (AArch64) reduces 4 lanes to scalar in one instruction
//   - dim=96 is divisible by 4 → 24 clean iterations, no scalar tail needed
//
// Platform: AArch64. NEON is mandatory on AArch64; no extra -march flag needed.
// Compile:  g++ main.cc -o main -O2 -fopenmp -lpthread -std=c++11
// =============================================================================
#pragma once
#include <arm_neon.h>
#include <queue>
#include <utility>
#include <cstdint>

// simd_inner_product_neon — dot product using one 128-bit Q register (4 lanes)
//
// One float32x4_t accumulator processes 4 floats per iteration.
// Horizontal reduce: vaddvq_f32 sums all 4 lanes into one scalar.
//
// Precondition: dim % 4 == 0  (DEEP100K: dim=96, 96÷4 = 24 iterations)
inline float simd_inner_product_neon(const float* a, const float* b, size_t dim)
{
    // sum: single 128-bit Q register accumulating 4 partial dot-products
    float32x4_t sum = vdupq_n_f32(0.0f);  // broadcast 0.0 to all 4 lanes

    for (size_t i = 0; i < dim; i += 4)
    {
        float32x4_t va = vld1q_f32(a + i);  //load
        float32x4_t vb = vld1q_f32(b + i);
        sum = vmlaq_f32(sum, va, vb); //FMA
    }
    // vaddvq_f32: horizontal sum
    return vaddvq_f32(sum);
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
    std::priority_queue<std::pair<float, uint32_t>> q;

    for (size_t i = 0; i < base_number; ++i)
    {
        float dis = 1.0f - simd_inner_product_neon(base + i * vecdim, query, vecdim);

        if (q.size() < k) {
            q.push({dis, (uint32_t)i});
        } else if (dis < q.top().first) {
            q.push({dis, (uint32_t)i});
            q.pop();
        }
    }
    return q;
}

// =============================================================================
// Optimized variant: 4-way loop unrolling with 4 independent accumulators
//
// Why faster than simd_flat_search:
//   vmlaq_f32 on AArch64 has ~4 cycle latency but 1/cycle throughput.
//   With 1 accumulator each iteration stalls 4 cycles waiting for the previous
//   FMA result — only 1 FMA in flight at a time.
//   With 4 independent accumulators the CPU issues 4 back-to-back FMAs with
//   no dependency between them, keeping the FMA pipeline fully occupied.
//   Processes 16 floats per iteration; dim=96 → 6 clean iterations, no tail.
// =============================================================================

// simd_inner_product_neon_unroll — 4x unrolled dot product, 4 independent accumulators
//
// Precondition: dim % 16 == 0  (DEEP100K: dim=96, 96÷16 = 6 iterations)
inline float simd_inner_product_neon_unroll(const float* a, const float* b, size_t dim)
{
    // 4 independent accumulators
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);

    // Unrolled 4x: 
    for (size_t i = 0; i < dim; i += 16)
    {
        // vld1q_f32: load 4 floats into one Q register
        // vmlaq_f32: sum += va * vb  (fused multiply-accumulate, 1 instruction)
        sum0 = vmlaq_f32(sum0, vld1q_f32(a + i),      vld1q_f32(b + i));
        sum1 = vmlaq_f32(sum1, vld1q_f32(a + i +  4), vld1q_f32(b + i +  4));
        sum2 = vmlaq_f32(sum2, vld1q_f32(a + i +  8), vld1q_f32(b + i +  8));
        sum3 = vmlaq_f32(sum3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    sum0 = vaddq_f32(sum0, sum1);
    sum2 = vaddq_f32(sum2, sum3);
    return vaddvq_f32(vaddq_f32(sum0, sum2));
}

// simd_flat_search_unroll — exhaustive k-NN using the unrolled inner product
std::priority_queue<std::pair<float, uint32_t>>
simd_flat_search_unroll(float* base, float* query,
                        size_t base_number, size_t vecdim, size_t k)
{
    std::priority_queue<std::pair<float, uint32_t>> q;

    for (size_t i = 0; i < base_number; ++i)
    {
        float dis = 1.0f - simd_inner_product_neon_unroll(base + i * vecdim, query, vecdim);

        if (q.size() < k) {
            q.push({dis, (uint32_t)i});
        } else if (dis < q.top().first) {
            q.push({dis, (uint32_t)i});
            q.pop();
        }
    }
    return q;
}
