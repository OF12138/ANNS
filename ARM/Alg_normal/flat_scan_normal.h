// =============================================================================
// flat_scan_normal.h — Scalar (non-SIMD) baseline for ARM flat k-NN scan
//
// This is the serial reference implementation used to measure speedup against
// SIMD-optimized versions in ARM/Alg_parallel/.
//
// Algorithm:
//   For each base vector i, compute IP distance = 1 - dot(base_i, query)
//   using a scalar loop, then maintain a max-heap of the k closest.
//
// Distance metric: Inner Product (IP), δ(x,y) = 1 − Σ xᵢ·yᵢ
//   DEEP100K vectors are normalized so IP distance ≡ cosine distance.
//
// This file mirrors the logic in the root flat_scan.h but is placed here
// as the ARM/Alg_normal entry point so the directory tree reflects the
// serial → SIMD progression used in the experiment report.
// =============================================================================
#pragma once
#include <queue>
#include <utility>
#include <cstdint>

// flat_search_normal — exhaustive scalar k-NN, no SIMD
//
// Identical logic to flat_search() in flat_scan.h.
// Return type matches the assignment requirement.
std::priority_queue<std::pair<float, uint32_t>>
flat_search_normal(float* base, float* query,
                   size_t base_number, size_t vecdim, size_t k)
{
    // Max-heap of (distance, index): top = farthest current candidate
    std::priority_queue<std::pair<float, uint32_t>> q;

    for (size_t i = 0; i < base_number; ++i) {
        float dis = 0.0f;

        // size_t counters produce 64-bit address arithmetic, which differs
        // from flat_scan.h's int counters and changes -O2 auto-vectorization.
        for (size_t d = 0; d < vecdim; ++d) {
            dis += base[d + i * vecdim] * query[d];
        }
        dis = 1.0f - dis;  // convert similarity → IP distance

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
