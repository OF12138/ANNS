// =============================================================================
// pq_flat_normal.h — Basic Product Quantization (PQ) flat k-NN scan
//
// Algorithm overview:
//   1. BUILD: split each d-dim vector into M subspaces of dsub = d/M dims.
//      Train K centroids per subspace via Lloyd's k-means (L2).
//      Encode every base vector as M uint8 codes (one centroid index per sub).
//
//   2. SEARCH (ADC — Asymmetric Distance Computation):
//      For each query, precompute a distance table: dtable[m][c] = IP(q_m, centroid[m][c]).
//      Scan all base codes: approx_IP(i) = Σ_m dtable[m][code[i,m]].
//      Return top-k by lowest approximate distance (1 − approx_IP).
//
// Default hyperparameters for DEEP100K (dim=96):
//   M=8  → dsub=12 dims per subspace
//   K=256 → 8-bit codes (one uint8 per subspace)
//
// Index memory: base_number × M bytes  (100K × 8 = ~800 KB for DEEP100K)
// Codebook memory: M × K × dsub floats (8 × 256 × 12 × 4 = ~96 KB)
// =============================================================================
#pragma once
#include <vector>
#include <queue>
#include <utility>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include <algorithm>

struct PQIndex 
{
    size_t M;           // number of subspaces
    size_t K;           // centroids per subspace (≤256 for uint8 codes)
    size_t dsub;        // dims per subspace = vecdim / M
    size_t vecdim;
    size_t base_number;

    // centroids[m * K * dsub + k * dsub + j]: centroid k of subspace m, dim j
    std::vector<float>   centroids;
    // codes[i * M + m]: centroid index for base vector i in subspace m
    std::vector<uint8_t> codes;

    void build(const float* base, size_t n, size_t d,
               size_t m_subs = 8, size_t k_cents = 256, int n_iters = 25);
};

// ---------------------------------------------------------------------------
// k-means for a single subspace (L2 distance).
// data: n × dsub row-major float matrix
// centroids (output): K × dsub, initialised from random samples on entry
// ---------------------------------------------------------------------------
static inline void pq_kmeans(const float* data, size_t n, size_t dsub,
                              size_t K, int n_iters, float* centroids)
{
    // Initialize centroids by sampling K vectors without replacement (LCG shuffle)
    std::vector<size_t> perm(n);
    for (size_t i = 0; i < n; ++i) perm[i] = i;
    unsigned rng = 0xA5A5A5A5u;
    for (size_t i = n - 1; i > 0; --i) {
        rng = rng * 1664525u + 1013904223u;
        size_t j = rng % (i + 1);
        size_t tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
    }
    for (size_t k = 0; k < K; ++k)
        memcpy(centroids + k * dsub, data + perm[k] * dsub, dsub * sizeof(float));

    std::vector<uint32_t> assign(n);
    std::vector<float>    cnt(K);

    for (int iter = 0; iter < n_iters; ++iter) {
        // Assign step: nearest centroid per vector (L2)
        for (size_t i = 0; i < n; ++i) {
            const float* v    = data + i * dsub;
            float        best = std::numeric_limits<float>::max();
            uint32_t     bk   = 0;
            for (size_t k = 0; k < K; ++k) {
                const float* c = centroids + k * dsub;
                float d2 = 0.0f;
                for (size_t j = 0; j < dsub; ++j) {
                    float diff = v[j] - c[j];
                    d2 += diff * diff;
                }
                if (d2 < best) { best = d2; bk = (uint32_t)k; }
            }
            assign[i] = bk;
        }

        // Update step: recompute centroids as cluster means
        memset(centroids, 0, K * dsub * sizeof(float));
        std::fill(cnt.begin(), cnt.end(), 0.0f);
        for (size_t i = 0; i < n; ++i) {
            uint32_t k   = assign[i];
            const float* v = data + i * dsub;
            float*       c = centroids + k * dsub;
            for (size_t j = 0; j < dsub; ++j) c[j] += v[j];
            cnt[k] += 1.0f;
        }
        for (size_t k = 0; k < K; ++k) {
            if (cnt[k] > 0.0f) {
                float inv = 1.0f / cnt[k];
                float* c  = centroids + k * dsub;
                for (size_t j = 0; j < dsub; ++j) c[j] *= inv;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PQIndex::build — trains M codebooks and encodes all base vectors
// ---------------------------------------------------------------------------
inline void PQIndex::build(const float* base, size_t n, size_t d,
                            size_t m_subs, size_t k_cents, int n_iters)
{
    M           = m_subs;
    K           = k_cents;
    dsub        = d / M;   // assumes d % M == 0
    vecdim      = d;
    base_number = n;

    centroids.resize(M * K * dsub, 0.0f);
    codes.resize(n * M);

    // Scratch buffer to hold one subspace of all base vectors
    std::vector<float> sub_data(n * dsub);

    for (size_t m = 0; m < M; ++m) 
    {
        // Extract subspace m from every base vector
        for (size_t i = 0; i < n; ++i)
            memcpy(sub_data.data() + i * dsub,
                   base + i * d + m * dsub,
                   dsub * sizeof(float));

        float* cents_m = centroids.data() + m * K * dsub;
        pq_kmeans(sub_data.data(), n, dsub, K, n_iters, cents_m);

        // Encode: assign each base vector to its nearest centroid
        for (size_t i = 0; i < n; ++i) 
        {
            const float* v    = sub_data.data() + i * dsub;
            float        best = std::numeric_limits<float>::max();
            uint8_t      bk   = 0;
            for (size_t k = 0; k < K; ++k) {
                const float* c = cents_m + k * dsub;
                float d2 = 0.0f;
                for (size_t j = 0; j < dsub; ++j) {
                    float diff = v[j] - c[j];
                    d2 += diff * diff;
                }
                if (d2 < best) { best = d2; bk = (uint8_t)k; }
            }
            codes[i * M + m] = bk;
        }
    }
}

// ---------------------------------------------------------------------------
// pq_flat_search_normal — ADC flat scan with precomputed IP distance table
//
// Step 1: dtable[m * K + c] = IP(query_m, centroid[m][c])
//         Computed once per query: M × K inner products over dsub dims each.
// Step 2: Scan all n codes; approx_IP(i) = Σ_m dtable[m * K + code[i,m]]
//         Return top-k as a max-heap of (distance, index) pairs.
// ---------------------------------------------------------------------------
inline std::priority_queue<std::pair<float, uint32_t>>
pq_flat_search_normal(const PQIndex& index, const float* query, size_t k)
{
    const size_t M    = index.M;
    const size_t K    = index.K;
    const size_t dsub = index.dsub;

    // Build distance table: dtable[m][c] = IP(q_m, centroid[m][c])
    std::vector<float> dtable(M * K);
    for (size_t m = 0; m < M; ++m) 
    {
        const float* q_m    = query + m * dsub;
        const float* c_base = index.centroids.data() + m * K * dsub;
        float*       row    = dtable.data() + m * K;
        for (size_t c = 0; c < K; ++c) 
        {
            const float* cv = c_base + c * dsub;
            float ip = 0.0f;
            for (size_t j = 0; j < dsub; ++j) ip += q_m[j] * cv[j];
            row[c] = ip;
        }
    }

    // Flat scan: sum K table lookups per base vector
    std::priority_queue<std::pair<float, uint32_t>> heap;
    for (size_t i = 0; i < index.base_number; ++i) 
    {
        const uint8_t* code = index.codes.data() + i * M;
        float approx_ip = 0.0f;
        for (size_t m = 0; m < M; ++m) approx_ip += dtable[m * K + code[m]];
        float dis = 1.0f - approx_ip;

        if (heap.size() < k)  heap.push({dis, static_cast<uint32_t>(i)});
        else if (dis < heap.top().first) 
        {
            heap.push({dis, static_cast<uint32_t>(i)});
            heap.pop();
        }
    }
    return heap;
}

// ---------------------------------------------------------------------------
// pq_flat_search_rerank — two-phase PQ k-NN search
//
// Phase 1 (coarse): same ADC scan as pq_flat_search_normal, keeps top-p
//                   candidates by approximate IP.
// Phase 2 (rerank): recomputes exact float32 IP against the original base
//                   vectors for those p candidates, returns top-k.
//
// The exact IP in Phase 2 recovers recall lost to quantization error, at the
// cost of p float32 dot products (vs. M table lookups for the coarse pass).
// ---------------------------------------------------------------------------
inline std::priority_queue<std::pair<float, uint32_t>>
pq_flat_search_rerank(const PQIndex& index, const float* base,
                      const float* query, size_t k, size_t p)
{
    const size_t M    = index.M;
    const size_t K    = index.K;
    const size_t dsub = index.dsub;
    const size_t d    = index.vecdim;

    // Build IP distance table once per query
    std::vector<float> dtable(M * K);
    for (size_t m = 0; m < M; ++m) 
    {
        const float* q_m    = query + m * dsub;
        const float* c_base = index.centroids.data() + m * K * dsub;
        float*       row    = dtable.data() + m * K;
        for (size_t c = 0; c < K; ++c) {
            const float* cv = c_base + c * dsub;
            float ip = 0.0f;
            for (size_t j = 0; j < dsub; ++j) ip += q_m[j] * cv[j];
            row[c] = ip;
        }
    }

    // Phase 1: coarse scan, keep top-p by approximate IP
    std::priority_queue<std::pair<float, uint32_t>> coarse_heap;
    for (size_t i = 0; i < index.base_number; ++i) {
        const uint8_t* code = index.codes.data() + i * M;
        float approx_ip = 0.0f;
        for (size_t m = 0; m < M; ++m) approx_ip += dtable[m * K + code[m]];
        float dis = 1.0f - approx_ip;

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

    // Phase 2: exact float32 IP rerank on the p candidates
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
