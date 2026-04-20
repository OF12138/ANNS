// =============================================================================
// sq_flat_normal.h — Scalar Quantization flat scan (scalar, no SIMD)
//
// Algorithm overview (two-phase):
//   Phase 1 — coarse scan: iterate over all base vectors in quantized (uint8)
//     form, compute an approximate IP distance, keep top-p candidates.
//   Phase 2 — rerank: recompute exact float32 IP distance for the p candidates,
//     return the best k.
//
// Why SQ is faster than plain float:
//   Each vector dimension is stored as 1 byte instead of 4 bytes (float32),
//   so the working set is 4× smaller → better L1/L2 cache utilisation.
//   uint8 → float widening + FMA has the same throughput as float in the
//   scalar path, but the 4× smaller footprint wins on memory-bound kernels.
//
// Quantization formula (per dimension d):
//   code[d] = round( (x[d] - min[d]) / scale[d] )   where scale = (max-min)/255
//   x[d] ≈ min[d] + code[d] * scale[d]
//
// Approximate dot product in coarse scan (no dequantization overhead):
//   dot(base_i, query) ≈ offset + Σ code[d] * adj_query[d]
//   where offset       = Σ min[d] * query[d]         (scalar, precomputed per query)
//         adj_query[d] = scale[d] * query[d]          (float, precomputed per query)
//
// Tradeoff parameter p:
//   Larger p → higher recall, higher latency.
//   p = base_number gives recall@k ≈ 1.0 (same as flat scan, but slower due to rerank).
//   p ≈ 100–500 gives good recall for DEEP100K at k=10.
// =============================================================================
#pragma once
#include <queue>
#include <utility>
#include <cstdint>
#include <vector>
#include <limits>
#include <cmath>

// SQIndex — stores per-dimension quantization parameters and uint8 codes
struct SQIndex 
{
    std::vector<uint8_t> codes;   // quantized base vectors, row-major [base_number × vecdim]
    std::vector<float>   mins;    // per-dimension minimum value [vecdim]
    std::vector<float>   scales;  // per-dimension scale = (max-min)/255  [vecdim]
    size_t base_number;
    size_t vecdim;

    // build — compute per-dimension min/scale and quantize all base vectors.
    // Must be called once before any search; takes O(n×d) time.
    void build(const float* base, size_t n, size_t d) 
    {
        base_number = n;
        vecdim      = d;
        mins.resize(d);
        scales.resize(d);
        codes.resize(n * d);

        // Pass 1: find per-dimension [min, max]
        for (size_t j = 0; j < d; ++j) 
        {
            float mn =  std::numeric_limits<float>::max();
            float mx = -std::numeric_limits<float>::max();
            for (size_t i = 0; i < n; ++i) 
            {
                float v = base[i * d + j];
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            mins[j]   = mn;
            // Avoid division by zero when all values are identical
            scales[j] = (mx > mn) ? (mx - mn) / 255.0f : 1.0f;
        }

        // Pass 2: quantize each dimension to uint8 via nearest-integer rounding
        for (size_t i = 0; i < n; ++i) 
        {
            for (size_t j = 0; j < d; ++j) 
            {
                float v = (base[i * d + j] - mins[j]) / scales[j];
                // Clamp to [0, 255] to guard against floating-point edge cases
                if (v < 0.0f)   v = 0.0f;
                if (v > 255.0f) v = 255.0f;
                codes[i * d + j] = static_cast<uint8_t>(v + 0.5f);
            }
        }
    }
};

// sq_flat_search_normal — two-phase SQ k-NN search, scalar (no SIMD)
//
// Parameters:
//   index  : pre-built SQIndex (build() must have been called)
//   base   : original float32 base vectors (needed for exact rerank)
//   query  : float32 query vector
//   k      : number of nearest neighbours to return
//   p      : coarse-scan candidate count; must satisfy p >= k
//
// Return: max-heap of (distance, index) pairs, matching flat_search signature.
std::priority_queue<std::pair<float, uint32_t>>
sq_flat_search_normal(const SQIndex& index, const float* base,
                      const float* query, size_t k, size_t p)
{
    const size_t d = index.vecdim;

    // -------------------------------------------------------------------------
    // Per-query precomputation:
    //   offset       = Σ min[j] * query[j]    (constant across all base vectors)
    //   adj_query[j] = scale[j] * query[j]    (absorbs the per-dim scale factor)
    //
    // This lets the inner loop become:
    //   approx_dot = offset + Σ code[j] * adj_query[j]
    // avoiding a multiply-add with min[] and scale[] inside the hot loop.
    // -------------------------------------------------------------------------
    std::vector<float> adj_query(d);
    float offset = 0.0f;
    for (size_t j = 0; j < d; ++j) {
        // adj_query[j] = scale[j] * query[j]: precompute so inner loop is just
        // uint8->float widening + FMA (no extra multiply by scale inside loop)
        adj_query[j] = index.scales[j] * query[j];
        offset += index.mins[j] * query[j];
    }

    // -------------------------------------------------------------------------
    // Phase 1 — coarse scan over uint8 codes (top-p candidates)
    //
    // Max-heap of (approx_distance, index): top = worst candidate so far.
    // We keep exactly p elements, evicting the farthest when the heap is full.
    // -------------------------------------------------------------------------
    std::priority_queue<std::pair<float, uint32_t>> coarse_heap;

    for (size_t i = 0; i < index.base_number; ++i) {
        // Approximate IP distance = 1 - approx_dot
        float dot = offset;
        const uint8_t* code = index.codes.data() + i * d;
        for (size_t j = 0; j < d; ++j) {
            // Cast uint8 to float before multiply — no integer overflow risk
            dot += static_cast<float>(code[j]) * adj_query[j];
        }
        float dis = 1.0f - dot;

        if (coarse_heap.size() < p) {
            coarse_heap.push({dis, static_cast<uint32_t>(i)});
        } else if (dis < coarse_heap.top().first) {
            // Replace the current worst candidate with this closer vector
            coarse_heap.push({dis, static_cast<uint32_t>(i)});
            coarse_heap.pop();
        }
    }

    // Extract candidate IDs from the heap
    std::vector<uint32_t> cand_ids;
    cand_ids.reserve(p);
    while (!coarse_heap.empty()) {
        cand_ids.push_back(coarse_heap.top().second);
        coarse_heap.pop();
    }

    // -------------------------------------------------------------------------
    // Phase 2 — exact rerank of p candidates using original float32 vectors
    //
    // Quantization introduces error; reranking corrects the ordering for the
    // top-p subset. This is what makes SQ approximate rather than exact:
    // a true nearest neighbour might fall outside the top-p coarse candidates.
    // -------------------------------------------------------------------------
    std::priority_queue<std::pair<float, uint32_t>> result;

    for (uint32_t id : cand_ids) {
        // Exact float32 inner product (same formula as flat_search_normal)
        float dot = 0.0f;
        const float* bv = base + static_cast<size_t>(id) * d;
        for (size_t j = 0; j < d; ++j) {
            dot += bv[j] * query[j];
        }
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
