// =============================================================================
// opq.h — Optimized Product Quantization (OPQ, non-parametric)
//
// Idea (Ge et al., "Optimized Product Quantization"): plain PQ cuts the space
// into M fixed, axis-aligned subspaces.  If the data's variance is unevenly
// distributed (or correlated) across those subspaces, some codebooks waste
// their 8 bits on low-information dimensions.  OPQ learns an orthogonal
// rotation R and quantizes R·x instead of x, minimizing
//
//     Σ_i || R x_i − q(R x_i) ||²        s.t.  RᵀR = I
//
// Alternating minimization (OPQ-NP):
//   (1) Fix R: refine the M codebooks with a few k-means iterations on R·X,
//       then re-encode → reconstructions Y (in rotated space).
//   (2) Fix codes: the optimal R solves the orthogonal Procrustes problem
//         max_R tr(R A),  A = Σ_i x_i y_iᵀ  (d×d)
//       whose solution is R = V Uᵀ from the SVD A = U S Vᵀ.
//   The 96×96 SVD is computed with one-sided Jacobi (Hestenes) in double
//   precision — no external LAPACK dependency, header-only.
//
// Search: rotation is a 96×96 mat-vec on the QUERY only (~1–2 µs with NEON).
// The coarse ADC scan runs on rotated-space codes with the rotated query's
// LUT; the exact rerank uses the ORIGINAL query against the ORIGINAL base,
// because R is orthogonal: IP(Rq, Rx) = IP(q, x).  All hot-path kernels are
// the existing cc_unroll LUT + gather scan — OPQ changes recall, not latency.
//
// Training cost is minutes (OMP-parallel offline); the trained index is
// cached under files/ like the other indices.
// =============================================================================
#pragma once
#include <arm_neon.h>
#include <vector>
#include <queue>
#include <utility>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <algorithm>
#include "../Alg_normal/pq_flat_normal.h"
#include "../Alg_parallel/pq_flat_simd.h"

struct OPQIndex
{
    size_t vecdim;
    std::vector<float> R;   // d×d row-major; rotated vector: (R x)[a] = Σ_b R[a*d+b] x[b]
    PQIndex pq;             // PQ trained on the rotated base
};

// ---------------------------------------------------------------------------
// Small NEON helpers (scalar tail handles len % 4 != 0)
// ---------------------------------------------------------------------------
static inline float opq_dot(const float* a, const float* b, size_t len)
{
    float32x4_t acc = vdupq_n_f32(0.0f);
    size_t j = 0;
    for (; j + 4 <= len; j += 4)
        acc = vmlaq_f32(acc, vld1q_f32(a + j), vld1q_f32(b + j));
    float s = vaddvq_f32(acc);
    for (; j < len; ++j) s += a[j] * b[j];
    return s;
}

static inline float opq_l2(const float* a, const float* b, size_t len)
{
    float32x4_t acc = vdupq_n_f32(0.0f);
    size_t j = 0;
    for (; j + 4 <= len; j += 4)
    {
        float32x4_t df = vsubq_f32(vld1q_f32(a + j), vld1q_f32(b + j));
        acc = vmlaq_f32(acc, df, df);
    }
    float s = vaddvq_f32(acc);
    for (; j < len; ++j)
    {
        float df = a[j] - b[j];
        s += df * df;
    }
    return s;
}

// ---------------------------------------------------------------------------
// opq_random_rotation — random orthonormal init via modified Gram-Schmidt
// (double precision internally; deterministic LCG for reproducibility).
// ---------------------------------------------------------------------------
inline void opq_random_rotation(size_t d, unsigned seed, std::vector<float>& R)
{
    std::vector<double> Rd(d * d);
    unsigned rng = seed;
    for (size_t t = 0; t < d * d; ++t)
    {
        rng = rng * 1664525u + 1013904223u;
        Rd[t] = (double)((rng >> 8) & 0xFFFFu) / 65536.0 * 2.0 - 1.0;
    }
    // Modified Gram-Schmidt on rows, two orthogonalization passes for stability
    for (size_t r = 0; r < d; ++r)
    {
        double* row = &Rd[r * d];
        for (int pass = 0; pass < 2; ++pass)
            for (size_t q = 0; q < r; ++q)
            {
                const double* prev = &Rd[q * d];
                double proj = 0.0;
                for (size_t j = 0; j < d; ++j) proj += row[j] * prev[j];
                for (size_t j = 0; j < d; ++j) row[j] -= proj * prev[j];
            }
        double nrm = 0.0;
        for (size_t j = 0; j < d; ++j) nrm += row[j] * row[j];
        nrm = std::sqrt(nrm);
        for (size_t j = 0; j < d; ++j) row[j] /= nrm;
    }
    R.resize(d * d);
    for (size_t t = 0; t < d * d; ++t) R[t] = (float)Rd[t];
}

// ---------------------------------------------------------------------------
// opq_rotate_all — out[i] = R · x[i] for all n vectors (OMP + NEON dot rows)
// ---------------------------------------------------------------------------
inline void opq_rotate_all(const float* R, const float* X, float* out,
                           size_t n, size_t d)
{
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < (long)n; ++i)
    {
        const float* x = X   + (size_t)i * d;
        float*       o = out + (size_t)i * d;
        for (size_t a = 0; a < d; ++a)
            o[a] = opq_dot(R + a * d, x, d);
    }
}

// ---------------------------------------------------------------------------
// opq_kmeans_iters — k-means refinement with GIVEN initial centroids
// (warm start; no re-initialization — pq_kmeans always re-inits, hence a new
// function).  data is strided: vector i of this subspace = data + i*stride.
// Empty clusters keep their previous centroid.
// ---------------------------------------------------------------------------
inline void opq_kmeans_iters(const float* data, size_t stride, size_t n,
                             size_t dsub, size_t K, int iters, float* centroids)
{
    std::vector<uint32_t> assign(n);
    std::vector<float>    cnt(K);
    std::vector<float>    old_cents(K * dsub);

    for (int iter = 0; iter < iters; ++iter)
    {
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < (long)n; ++i)
        {
            const float* v    = data + (size_t)i * stride;
            float        best = std::numeric_limits<float>::max();
            uint32_t     bk   = 0;
            for (size_t c = 0; c < K; ++c)
            {
                float d2 = opq_l2(v, centroids + c * dsub, dsub);
                if (d2 < best) { best = d2; bk = (uint32_t)c; }
            }
            assign[i] = bk;
        }

        memcpy(old_cents.data(), centroids, K * dsub * sizeof(float));
        memset(centroids, 0, K * dsub * sizeof(float));
        std::fill(cnt.begin(), cnt.end(), 0.0f);
        for (size_t i = 0; i < n; ++i)
        {
            const float* v = data + i * stride;
            float*       c = centroids + assign[i] * dsub;
            for (size_t j = 0; j < dsub; ++j) c[j] += v[j];
            cnt[assign[i]] += 1.0f;
        }
        for (size_t c = 0; c < K; ++c)
        {
            if (cnt[c] > 0.0f)
            {
                float  inv = 1.0f / cnt[c];
                float* cc  = centroids + c * dsub;
                for (size_t j = 0; j < dsub; ++j) cc[j] *= inv;
            }
            else
            {
                memcpy(centroids + c * dsub, old_cents.data() + c * dsub,
                       dsub * sizeof(float));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// opq_encode_all — encode every rotated vector; returns total squared
// quantization error (the OPQ objective, printed per outer iteration).
// ---------------------------------------------------------------------------
inline double opq_encode_all(const float* xrot, size_t n, size_t d,
                             size_t M, size_t K, size_t dsub,
                             const float* centroids, uint8_t* codes)
{
    double err = 0.0;
    #pragma omp parallel for schedule(static) reduction(+:err)
    for (long i = 0; i < (long)n; ++i)
    {
        const float* v = xrot + (size_t)i * d;
        for (size_t m = 0; m < M; ++m)
        {
            const float* v_m     = v + m * dsub;
            const float* cents_m = centroids + m * K * dsub;
            float        best    = std::numeric_limits<float>::max();
            uint8_t      bk      = 0;
            for (size_t c = 0; c < K; ++c)
            {
                float d2 = opq_l2(v_m, cents_m + c * dsub, dsub);
                if (d2 < best) { best = d2; bk = (uint8_t)c; }
            }
            codes[(size_t)i * M + m] = bk;
            err += best;
        }
    }
    return err;
}

// ---------------------------------------------------------------------------
// opq_reconstruct_all — decode codes back to rotated-space reconstructions Y
// ---------------------------------------------------------------------------
inline void opq_reconstruct_all(const uint8_t* codes, const float* centroids,
                                size_t n, size_t d, size_t M, size_t K,
                                size_t dsub, float* Y)
{
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < (long)n; ++i)
        for (size_t m = 0; m < M; ++m)
            memcpy(Y + (size_t)i * d + m * dsub,
                   centroids + m * K * dsub
                             + (size_t)codes[(size_t)i * M + m] * dsub,
                   dsub * sizeof(float));
}

// ---------------------------------------------------------------------------
// opq_svd_jacobi — one-sided Jacobi (Hestenes) SVD of a d×d matrix.
// Input A row-major.  Output: Ucol / Vcol are d×d COLUMN-major with
// orthonormal columns such that A = U S Vᵀ (S not returned; R = V Uᵀ only
// needs the two orthogonal factors).
// ---------------------------------------------------------------------------
inline void opq_svd_jacobi(const std::vector<double>& A, size_t d,
                           std::vector<double>& Ucol, std::vector<double>& Vcol)
{
    std::vector<double>& G = Ucol;          // G starts as A's columns → becomes U·S
    G.assign(d * d, 0.0);
    Vcol.assign(d * d, 0.0);
    for (size_t i = 0; i < d; ++i)
        for (size_t j = 0; j < d; ++j)
            G[j * d + i] = A[i * d + j];    // column j of A
    for (size_t j = 0; j < d; ++j) Vcol[j * d + j] = 1.0;

    const double tol = 1e-11;
    for (int sweep = 0; sweep < 60; ++sweep)
    {
        int rotated = 0;
        for (size_t j1 = 0; j1 + 1 < d; ++j1)
            for (size_t j2 = j1 + 1; j2 < d; ++j2)
            {
                double* g1 = &G[j1 * d];
                double* g2 = &G[j2 * d];
                double alpha = 0.0, beta = 0.0, gamma = 0.0;
                for (size_t i = 0; i < d; ++i)
                {
                    alpha += g1[i] * g1[i];
                    beta  += g2[i] * g2[i];
                    gamma += g1[i] * g2[i];
                }
                if (std::fabs(gamma) <= tol * std::sqrt(alpha * beta)) continue;

                // Jacobi rotation zeroing the off-diagonal of [[α,γ],[γ,β]]
                double zeta = (beta - alpha) / (2.0 * gamma);
                double t    = ((zeta >= 0.0) ? 1.0 : -1.0)
                              / (std::fabs(zeta) + std::sqrt(1.0 + zeta * zeta));
                double c    = 1.0 / std::sqrt(1.0 + t * t);
                double s    = c * t;

                double* v1 = &Vcol[j1 * d];
                double* v2 = &Vcol[j2 * d];
                for (size_t i = 0; i < d; ++i)
                {
                    double a1 = g1[i], a2 = g2[i];
                    g1[i] = c * a1 - s * a2;
                    g2[i] = s * a1 + c * a2;
                    double b1 = v1[i], b2 = v2[i];
                    v1[i] = c * b1 - s * b2;
                    v2[i] = s * b1 + c * b2;
                }
                ++rotated;
            }
        if (rotated == 0) break;
    }

    // Normalize columns of G → U  (degenerate columns are practically
    // unreachable for real data; guarded with a basis fallback anyway)
    for (size_t j = 0; j < d; ++j)
    {
        double* g   = &G[j * d];
        double  nrm = 0.0;
        for (size_t i = 0; i < d; ++i) nrm += g[i] * g[i];
        nrm = std::sqrt(nrm);
        if (nrm > 1e-12)
            for (size_t i = 0; i < d; ++i) g[i] /= nrm;
        else
        {
            for (size_t i = 0; i < d; ++i) g[i] = 0.0;
            g[j] = 1.0;
        }
    }
}

// ---------------------------------------------------------------------------
// opq_procrustes — R = argmax_{RᵀR=I} tr(R A), A[a][b] = Σ_i X[i][a]·Y[i][b].
// A = U S Vᵀ  →  R = V Uᵀ.  Accumulation in double (n = 100K terms).
// ---------------------------------------------------------------------------
inline void opq_procrustes(const float* X, const float* Y, size_t n, size_t d,
                           float* R)
{
    std::vector<double> A(d * d, 0.0);
    #pragma omp parallel
    {
        std::vector<double> loc(d * d, 0.0);
        #pragma omp for nowait schedule(static)
        for (long i = 0; i < (long)n; ++i)
        {
            const float* x = X + (size_t)i * d;
            const float* y = Y + (size_t)i * d;
            for (size_t a = 0; a < d; ++a)
            {
                double  xa  = (double)x[a];
                double* row = &loc[a * d];
                for (size_t b = 0; b < d; ++b) row[b] += xa * (double)y[b];
            }
        }
        #pragma omp critical
        for (size_t t = 0; t < d * d; ++t) A[t] += loc[t];
    }

    std::vector<double> Ucol, Vcol;
    opq_svd_jacobi(A, d, Ucol, Vcol);

    // R[a][b] = Σ_j V[a][j] · U[b][j]   (columns j of the column-major factors)
    for (size_t a = 0; a < d; ++a)
        for (size_t b = 0; b < d; ++b)
        {
            double acc = 0.0;
            for (size_t j = 0; j < d; ++j)
                acc += Vcol[j * d + a] * Ucol[j * d + b];
            R[a * d + b] = (float)acc;
        }
}

// ---------------------------------------------------------------------------
// opq_build — alternating minimization + final full PQ build on rotated data.
// Prints per-iteration distortion to stderr (report material: the objective
// must decrease monotonically; plain PQ = iteration 0 with R = random).
// ---------------------------------------------------------------------------
inline void opq_build(OPQIndex& out, const float* base, size_t n, size_t d,
                      size_t M = 8, size_t K = 256, int opq_iters = 8,
                      int kmeans_inner = 3, int kmeans_final = 25)
{
    out.vecdim = d;
    const size_t dsub = d / M;

    opq_random_rotation(d, 0x2411264u, out.R);

    std::vector<float>   xrot((size_t)n * d);
    std::vector<float>   recon((size_t)n * d);
    std::vector<float>   cents(M * K * dsub);
    std::vector<uint8_t> codes((size_t)n * M);

    opq_rotate_all(out.R.data(), base, xrot.data(), n, d);

    // Initial codebooks: sample K rotated vectors per subspace (LCG shuffle)
    for (size_t m = 0; m < M; ++m)
    {
        unsigned rng = 0xA5A5A5A5u + (unsigned)m * 2654435761u;
        for (size_t c = 0; c < K; ++c)
        {
            rng = rng * 1664525u + 1013904223u;
            size_t pick = (size_t)(rng % n);
            memcpy(cents.data() + (m * K + c) * dsub,
                   xrot.data() + pick * d + m * dsub,
                   dsub * sizeof(float));
        }
    }

    for (int it = 0; it < opq_iters; ++it)
    {
        // (1) Fix R: refine codebooks on the current rotation, re-encode
        for (size_t m = 0; m < M; ++m)
            opq_kmeans_iters(xrot.data() + m * dsub, d, n, dsub, K,
                             kmeans_inner, cents.data() + m * K * dsub);

        double err = opq_encode_all(xrot.data(), n, d, M, K, dsub,
                                    cents.data(), codes.data());
        std::cerr << "[opq] iter " << it
                  << "  distortion/vec = " << err / (double)n << "\n";

        // (2) Fix codes: Procrustes update of R, then re-rotate the base
        opq_reconstruct_all(codes.data(), cents.data(), n, d, M, K, dsub,
                            recon.data());
        opq_procrustes(base, recon.data(), n, d, out.R.data());
        opq_rotate_all(out.R.data(), base, xrot.data(), n, d);
    }

    // Sanity: report max |R Rᵀ − I| (should be ~1e-6 in float)
    {
        double worst = 0.0;
        for (size_t a = 0; a < d; ++a)
            for (size_t b = 0; b < d; ++b)
            {
                double acc = 0.0;
                for (size_t j = 0; j < d; ++j)
                    acc += (double)out.R[a * d + j] * (double)out.R[b * d + j];
                double dev = std::fabs(acc - (a == b ? 1.0 : 0.0));
                if (dev > worst) worst = dev;
            }
        std::cerr << "[opq] R orthogonality max|RRt-I| = " << worst << "\n";
    }

    // Final: full PQ build (fresh init, full iterations) on the final rotation
    pq_build_index_simd_blocked(out.pq, xrot.data(), n, d, M, K, kmeans_final);
}

// ---------------------------------------------------------------------------
// opq_save / opq_load — binary cache under files/ (build takes minutes)
// ---------------------------------------------------------------------------
inline bool opq_save(const OPQIndex& idx, const char* path)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    uint64_t hdr[5] = { (uint64_t)idx.vecdim, (uint64_t)idx.pq.M,
                        (uint64_t)idx.pq.K,   (uint64_t)idx.pq.dsub,
                        (uint64_t)idx.pq.base_number };
    f.write((const char*)hdr, sizeof(hdr));
    f.write((const char*)idx.R.data(),            idx.R.size() * sizeof(float));
    f.write((const char*)idx.pq.centroids.data(), idx.pq.centroids.size() * sizeof(float));
    f.write((const char*)idx.pq.codes.data(),     idx.pq.codes.size());
    return f.good();
}

inline bool opq_load(OPQIndex& idx, const char* path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    uint64_t hdr[5];
    f.read((char*)hdr, sizeof(hdr));
    if (!f.good()) return false;
    idx.vecdim         = (size_t)hdr[0];
    idx.pq.M           = (size_t)hdr[1];
    idx.pq.K           = (size_t)hdr[2];
    idx.pq.dsub        = (size_t)hdr[3];
    idx.pq.base_number = (size_t)hdr[4];
    idx.pq.vecdim      = idx.vecdim;
    idx.R.resize(idx.vecdim * idx.vecdim);
    idx.pq.centroids.resize(idx.pq.M * idx.pq.K * idx.pq.dsub);
    idx.pq.codes.resize(idx.pq.base_number * idx.pq.M);
    f.read((char*)idx.R.data(),            idx.R.size() * sizeof(float));
    f.read((char*)idx.pq.centroids.data(), idx.pq.centroids.size() * sizeof(float));
    f.read((char*)idx.pq.codes.data(),     idx.pq.codes.size());
    return f.good();
}

// ---------------------------------------------------------------------------
// opq_search_gather — rotate the query, reuse the existing best PQ hot path
// (cc_unroll LUT + gather coarse scan), rerank with the ORIGINAL query and
// base (valid because IP(Rq, Rx) = IP(q, x) for orthogonal R).
// Precondition: M == 8 (gather scan requirement, same as SEARCH_ALG 13).
// ---------------------------------------------------------------------------
inline std::priority_queue<std::pair<float, uint32_t>>
opq_search_gather(const OPQIndex& opq, const PQIndexSIMD& pq_simd,
                  const float* base, const float* query, size_t k, size_t p)
{
    const size_t d = opq.vecdim;
    std::vector<float> qrot(d);
    for (size_t a = 0; a < d; ++a)
        qrot[a] = opq_dot(opq.R.data() + a * d, query, d);

    std::vector<float> dtable(opq.pq.M * opq.pq.K);
    pq_build_lut_cc_unroll(pq_simd, qrot.data(), dtable.data());
    return pq_rerank_from_dtable_gather(opq.pq, base, query, k, p, dtable.data());
}
