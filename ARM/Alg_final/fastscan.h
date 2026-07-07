// =============================================================================
// fastscan.h — PQ4 FastScan: 4-bit PQ with a register-resident LUT (NEON vqtbl1q)
//
// Motivation (Andre et al., "Cache locality is not enough: high-performance
// nearest neighbor search with product quantization fast scan"):
//   The classic PQ ADC scan is bound by LUT lookups that go through the cache
//   hierarchy (L1 at best).  FastScan shrinks each sub-quantizer codebook to
//   K=16 centroids so that one LUT row (16 × uint8 = 16 bytes) fits in a single
//   128-bit NEON register, and replaces the memory lookup with the in-register
//   byte-shuffle instruction vqtbl1q_u8: 16 table lookups per instruction.
//
// Configuration for DEEP100K (dim = 96):
//   M = 16 subspaces, K = 16 centroids, dsub = 6.
//   Code size per vector = 16 × 4 bit = 8 bytes — identical to the course PQ
//   baseline (M=8, K=256), so recall differences isolate the "many coarse
//   subquantizers vs few fine ones" trade-off at equal memory.
//
// Storage layout (blocks of 32 vectors):
//   Two neighbouring subspaces (2m, 2m+1) are packed into one byte:
//     byte = code[2m] | (code[2m+1] << 4)
//   Per block b, per subspace pair pm, 32 consecutive bytes hold the packed
//   codes of the 32 block vectors (16 bytes for vectors 0..15, 16 for 16..31):
//     packed[b*(M*16) + pm*32 + h*16 + j]   (h = half, j = byte)
//
// Scan kernel (per block of 32 vectors):
//   For each pair pm: load 16B packed codes, split nibbles (vandq / vshrq_n),
//   two vqtbl1q_u8 lookups against the two 16-byte LUT rows, widen-accumulate
//   into four uint16x8 accumulators.  Max sum = 16 × 255 = 4080 < 65535, so
//   plain (non-saturating) uint16 accumulation is exact.
//
// LUT quantization (per query):
//   lut_f[m][c] = −IP(q_m, centroid[m][c])   (lower = better)
//   per-row bias b_m = min_c lut_f[m][c]; one shared scale
//   s = 255 / max_m(max_c − min_c);  qlut[m][c] = round((lut_f − b_m) · s).
//   Per-row bias + shared scale is an order-preserving affine map of the sum,
//   so the integer ranking equals the float ranking up to quantization error.
//   The top-p candidates are re-ranked with exact float32 IP anyway.
// =============================================================================
#pragma once
#include <arm_neon.h>
#include <vector>
#include <queue>
#include <utility>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include <algorithm>
#include "../Alg_normal/pq_flat_normal.h"   // pq_kmeans

struct FastScanIndex
{
    size_t M;            // subspaces (16)
    size_t K;            // centroids per subspace (fixed 16 → 4-bit codes)
    size_t dsub;         // dims per subspace = vecdim / M
    size_t vecdim;
    size_t base_number;
    size_t nblocks;      // ceil(base_number / 32)

    // centroids[m * K * dsub + c * dsub + j] — same layout as PQIndex
    std::vector<float>   centroids;
    // codes[i * M + m] ∈ [0,16) — unpacked, used by the scalar baseline + packing
    std::vector<uint8_t> codes;
    // nibble-packed block layout, nblocks * M * 16 bytes (= 8 B / vector)
    std::vector<uint8_t> packed;
};

// ---------------------------------------------------------------------------
// fastscan_build — train M 16-centroid codebooks, encode, pack into blocks.
// Subspaces are independent → parallelised with OpenMP (offline phase).
// ---------------------------------------------------------------------------
inline void fastscan_build(FastScanIndex& out, const float* base, size_t n, size_t d,
                           size_t m_subs = 16, int n_iters = 25)
{
    out.M           = m_subs;
    out.K           = 16;
    out.dsub        = d / m_subs;    // requires d % m_subs == 0
    out.vecdim      = d;
    out.base_number = n;

    const size_t M    = out.M;
    const size_t K    = out.K;
    const size_t dsub = out.dsub;

    out.centroids.assign(M * K * dsub, 0.0f);
    out.codes.assign(n * M, 0);

    #pragma omp parallel for schedule(dynamic)
    for (int m = 0; m < (int)M; ++m)
    {
        // Extract subspace m of every base vector (contiguous copy for k-means)
        std::vector<float> sub_data(n * dsub);
        for (size_t i = 0; i < n; ++i)
            memcpy(sub_data.data() + i * dsub,
                   base + i * d + (size_t)m * dsub,
                   dsub * sizeof(float));

        float* cents_m = out.centroids.data() + (size_t)m * K * dsub;
        pq_kmeans(sub_data.data(), n, dsub, K, n_iters, cents_m);

        // Encode: nearest centroid (L2) per vector
        for (size_t i = 0; i < n; ++i)
        {
            const float* v    = sub_data.data() + i * dsub;
            float        best = std::numeric_limits<float>::max();
            uint8_t      bk   = 0;
            for (size_t c = 0; c < K; ++c)
            {
                const float* cv = cents_m + c * dsub;
                float d2 = 0.0f;
                for (size_t j = 0; j < dsub; ++j)
                {
                    float diff = v[j] - cv[j];
                    d2 += diff * diff;
                }
                if (d2 < best) { best = d2; bk = (uint8_t)c; }
            }
            out.codes[i * M + m] = bk;
        }
    }

    // Pack into 32-vector blocks: byte = code[2pm] | code[2pm+1] << 4.
    // Tail vectors of the last block are padded with code 0; the scan loop
    // never pushes ids ≥ n, so padding cannot enter the candidate set.
    out.nblocks = (n + 31) / 32;
    out.packed.assign(out.nblocks * M * 16, 0);
    for (size_t b = 0; b < out.nblocks; ++b)
        for (size_t pm = 0; pm < M / 2; ++pm)
            for (size_t h = 0; h < 2; ++h)
                for (size_t j = 0; j < 16; ++j)
                {
                    size_t v = b * 32 + h * 16 + j;
                    if (v >= n) continue;
                    uint8_t lo = out.codes[v * M + 2 * pm];
                    uint8_t hi = out.codes[v * M + 2 * pm + 1];
                    out.packed[b * (M * 16) + pm * 32 + h * 16 + j]
                        = (uint8_t)(lo | (hi << 4));
                }
}

// ---------------------------------------------------------------------------
// fastscan_build_qlut — per-query float LUT + 8-bit quantization.
// lut_f[m][c] = −IP(q_m, centroid) so that SMALLER quantized sum = better.
// Per-row bias, one shared scale (order-preserving over the sum).
// Precondition: M * K ≤ 512 (M=16, K=16 → 256 ✓)
// ---------------------------------------------------------------------------
inline void fastscan_build_qlut(const FastScanIndex& fs, const float* query,
                                uint8_t* qlut /* M*K bytes */)
{
    const size_t M    = fs.M;
    const size_t K    = fs.K;
    const size_t dsub = fs.dsub;

    float lut_f[512];
    float bias[32];
    float delta_max = 0.0f;

    for (size_t m = 0; m < M; ++m)
    {
        const float* q_m    = query + m * dsub;
        const float* c_base = fs.centroids.data() + m * K * dsub;
        float row_min =  std::numeric_limits<float>::max();
        float row_max = -std::numeric_limits<float>::max();
        for (size_t c = 0; c < K; ++c)
        {
            const float* cv = c_base + c * dsub;
            float ip = 0.0f;
            for (size_t j = 0; j < dsub; ++j) ip += q_m[j] * cv[j];
            float v = -ip;                       // lower = better
            lut_f[m * K + c] = v;
            if (v < row_min) row_min = v;
            if (v > row_max) row_max = v;
        }
        bias[m] = row_min;
        if (row_max - row_min > delta_max) delta_max = row_max - row_min;
    }

    const float scale = (delta_max > 0.0f) ? 255.0f / delta_max : 0.0f;
    for (size_t m = 0; m < M; ++m)
        for (size_t c = 0; c < K; ++c)
        {
            float q = (lut_f[m * K + c] - bias[m]) * scale + 0.5f;
            if (q > 255.0f) q = 255.0f;
            qlut[m * K + c] = (uint8_t)q;
        }
}

// ---------------------------------------------------------------------------
// fastscan_scan_block_neon — the FastScan kernel: 32 vectors per call.
// Per subspace pair: 2 loads, 2 nibble-splits, 4 vqtbl1q_u8 (64 lookups),
// 8 widening adds.  The LUT never leaves the register file.
// ---------------------------------------------------------------------------
static inline void fastscan_scan_block_neon(const uint8_t* blk, const uint8_t* qlut,
                                            size_t M, uint16_t* sums /* 32 */)
{
    uint16x8_t acc0 = vdupq_n_u16(0);   // vectors  0..7
    uint16x8_t acc1 = vdupq_n_u16(0);   // vectors  8..15
    uint16x8_t acc2 = vdupq_n_u16(0);   // vectors 16..23
    uint16x8_t acc3 = vdupq_n_u16(0);   // vectors 24..31
    const uint8x16_t nib_mask = vdupq_n_u8(0x0F);

    const uint8_t* ptr = blk;
    for (size_t pm = 0; pm < M / 2; ++pm)
    {
        // Two LUT rows live in registers for the whole pair
        const uint8x16_t lut_even = vld1q_u8(qlut + (2 * pm)     * 16);
        const uint8x16_t lut_odd  = vld1q_u8(qlut + (2 * pm + 1) * 16);

        uint8x16_t packed0 = vld1q_u8(ptr);        // vectors 0..15
        uint8x16_t packed1 = vld1q_u8(ptr + 16);   // vectors 16..31
        ptr += 32;

        // 4 × 16 in-register lookups
        uint8x16_t v0e = vqtbl1q_u8(lut_even, vandq_u8(packed0, nib_mask));
        uint8x16_t v0o = vqtbl1q_u8(lut_odd,  vshrq_n_u8(packed0, 4));
        uint8x16_t v1e = vqtbl1q_u8(lut_even, vandq_u8(packed1, nib_mask));
        uint8x16_t v1o = vqtbl1q_u8(lut_odd,  vshrq_n_u8(packed1, 4));

        // Widening accumulate uint8 → uint16 (exact: max 16 × 255 = 4080)
        acc0 = vaddw_u8(acc0, vget_low_u8 (v0e));
        acc0 = vaddw_u8(acc0, vget_low_u8 (v0o));
        acc1 = vaddw_u8(acc1, vget_high_u8(v0e));
        acc1 = vaddw_u8(acc1, vget_high_u8(v0o));
        acc2 = vaddw_u8(acc2, vget_low_u8 (v1e));
        acc2 = vaddw_u8(acc2, vget_low_u8 (v1o));
        acc3 = vaddw_u8(acc3, vget_high_u8(v1e));
        acc3 = vaddw_u8(acc3, vget_high_u8(v1o));
    }
    vst1q_u16(sums,      acc0);
    vst1q_u16(sums +  8, acc1);
    vst1q_u16(sums + 16, acc2);
    vst1q_u16(sums + 24, acc3);
}

// ---------------------------------------------------------------------------
// fastscan_coarse_topp — quantized-LUT block scan, keeps top-p candidate ids.
// ---------------------------------------------------------------------------
inline void fastscan_coarse_topp(const FastScanIndex& fs, const float* query,
                                 size_t p, std::vector<uint32_t>& cand_ids)
{
    uint8_t qlut[512];
    fastscan_build_qlut(fs, query, qlut);

    const size_t n  = fs.base_number;
    const size_t nb = fs.nblocks;
    const size_t M  = fs.M;

    // Max-heap on the quantized distance keeps the p smallest sums.
    std::priority_queue<std::pair<float, uint32_t>> heap;
    uint16_t sums[32];

    for (size_t b = 0; b < nb; ++b)
    {
        fastscan_scan_block_neon(fs.packed.data() + b * (M * 16), qlut, M, sums);
        const size_t lim = (n - b * 32 < 32) ? (n - b * 32) : 32;
        for (size_t j = 0; j < lim; ++j)
        {
            float    dis = (float)sums[j];
            uint32_t id  = (uint32_t)(b * 32 + j);
            if (heap.size() < p)
            {
                heap.push({dis, id});
            }
            else if (dis < heap.top().first)
            {
                heap.push({dis, id});
                heap.pop();
            }
        }
    }

    cand_ids.clear();
    cand_ids.reserve(p);
    while (!heap.empty())
    {
        cand_ids.push_back(heap.top().second);
        heap.pop();
    }
}

// ---------------------------------------------------------------------------
// pq4_coarse_topp — scalar ADC baseline over the SAME 4-bit codes (float LUT,
// per-vector M lookups through the cache).  Isolates the kernel effect:
// pq4_scalar vs fastscan differ ONLY in how the LUT is looked up.
// ---------------------------------------------------------------------------
inline void pq4_coarse_topp(const FastScanIndex& fs, const float* query,
                            size_t p, std::vector<uint32_t>& cand_ids)
{
    const size_t M    = fs.M;
    const size_t K    = fs.K;
    const size_t dsub = fs.dsub;
    const size_t n    = fs.base_number;

    float dtable[512];
    for (size_t m = 0; m < M; ++m)
    {
        const float* q_m    = query + m * dsub;
        const float* c_base = fs.centroids.data() + m * K * dsub;
        for (size_t c = 0; c < K; ++c)
        {
            const float* cv = c_base + c * dsub;
            float ip = 0.0f;
            for (size_t j = 0; j < dsub; ++j) ip += q_m[j] * cv[j];
            dtable[m * K + c] = ip;
        }
    }

    std::priority_queue<std::pair<float, uint32_t>> heap;
    for (size_t i = 0; i < n; ++i)
    {
        const uint8_t* code = fs.codes.data() + i * M;
        float approx_ip = 0.0f;
        for (size_t m = 0; m < M; ++m) approx_ip += dtable[m * K + code[m]];
        float dis = 1.0f - approx_ip;

        if (heap.size() < p)
        {
            heap.push({dis, (uint32_t)i});
        }
        else if (dis < heap.top().first)
        {
            heap.push({dis, (uint32_t)i});
            heap.pop();
        }
    }

    cand_ids.clear();
    cand_ids.reserve(p);
    while (!heap.empty())
    {
        cand_ids.push_back(heap.top().second);
        heap.pop();
    }
}

// ---------------------------------------------------------------------------
// fastscan_rerank_exact — exact float32 IP rerank of the candidate list
// (identical style to pq_rerank_from_dtable's Phase 2).
// ---------------------------------------------------------------------------
static inline std::priority_queue<std::pair<float, uint32_t>>
fastscan_rerank_exact(const float* base, const float* query, size_t d, size_t k,
                      const std::vector<uint32_t>& cand_ids)
{
    std::priority_queue<std::pair<float, uint32_t>> result;
    for (size_t idx = 0; idx < cand_ids.size(); ++idx)
    {
        uint32_t     id = cand_ids[idx];
        const float* bv = base + (size_t)id * d;
        float ip = 0.0f;
        for (size_t j = 0; j < d; ++j) ip += bv[j] * query[j];
        float dis = 1.0f - ip;

        if (result.size() < k)
        {
            result.push({dis, id});
        }
        else if (dis < result.top().first)
        {
            result.push({dis, id});
            result.pop();
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// pq4_search_rerank — two-phase search: scalar 4-bit ADC coarse + exact rerank
// ---------------------------------------------------------------------------
inline std::priority_queue<std::pair<float, uint32_t>>
pq4_search_rerank(const FastScanIndex& fs, const float* base,
                  const float* query, size_t k, size_t p)
{
    std::vector<uint32_t> cand_ids;
    pq4_coarse_topp(fs, query, p, cand_ids);
    return fastscan_rerank_exact(base, query, fs.vecdim, k, cand_ids);
}

// ---------------------------------------------------------------------------
// fastscan_search_rerank — two-phase search: FastScan coarse + exact rerank
// ---------------------------------------------------------------------------
inline std::priority_queue<std::pair<float, uint32_t>>
fastscan_search_rerank(const FastScanIndex& fs, const float* base,
                       const float* query, size_t k, size_t p)
{
    std::vector<uint32_t> cand_ids;
    fastscan_coarse_topp(fs, query, p, cand_ids);
    return fastscan_rerank_exact(base, query, fs.vecdim, k, cand_ids);
}
