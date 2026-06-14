// =============================================================================
// gemm_knn.cu -- Brute-force GEMM kNN baseline (Manual §3, direction 1 baseline)
//
// Idea: inner-product distance = row(query) · column(base).  So the entire
// distance computation for a batch of queries is one matrix multiply:
//
//     S = Q (m x d) · Bᵀ (d x n)  ->  S (m x n),  S[i][j] = IP(query_i, base_j)
//
// Then for each query (row of S) we take the top-k LARGEST inner products
// (DEEP vectors are unit-normalized, so larger IP = nearer neighbor).
//
// This file is the MOST BASIC baseline:
//   - distance kernel: one thread per (query, base) pair, naive d-length dot.
//   - top-k kernel:    one thread per query, linear scan + insertion into a
//                      length-k array.
// No tiling, no shared memory, no cuBLAS.  Those come as later variants so the
// speedups can be measured against this reference (CLAUDE.md: keep the original).
//
// The m x n score matrix is 100000 x 10000 x 4 B = 4 GB, too big for 8 GB at
// once alongside everything else, so queries are processed in CHUNKS.
//
// Build:  nvcc gemm_knn.cu -o gemm_knn -O2 -arch=sm_89
// Run  :  ./gemm_knn            (expects ./data/DEEP100K.*)
// =============================================================================

#include "gpu_utils.cuh"
#include <cfloat>

#define DIM        96      // DEEP100K vector dimension
#define K          10      // neighbors per query
#define QCHUNK     2048    // queries processed per GPU pass (tunable)
#define TOPK_TPB   128     // threads per block for top-k kernel


// ── Distance kernel: S[i][j] = IP(query_i, base_j) ───────────────────────────
// Grid is 2D: x over base (n), y over queries-in-chunk (qn).
// Each thread computes one full d-length dot product.
__global__ void ip_distance_kernel(const float* __restrict__ Q,
                                    const float* __restrict__ B,
                                    float* __restrict__ S,
                                    int qn, int n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;   // base index
    int i = blockIdx.y * blockDim.y + threadIdx.y;   // query index (within chunk)
    if (i >= qn || j >= n) return;

    const float* q = Q + (size_t)i * DIM;
    const float* b = B + (size_t)j * DIM;
    float acc = 0.0f;
    #pragma unroll
    for (int t = 0; t < DIM; ++t)
        acc += q[t] * b[t];

    S[(size_t)i * n + j] = acc;
}


// ── Top-k kernel: per query row, keep K largest IPs (insertion into sorted) ──
__global__ void topk_kernel(const float* __restrict__ S,
                            int qn, int n,
                            int32_t* __restrict__ out_ids,
                            float*   __restrict__ out_val)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // query index (within chunk)
    if (i >= qn) return;

    float bestVal[K];
    int   bestId [K];
    #pragma unroll
    for (int t = 0; t < K; ++t) { bestVal[t] = -FLT_MAX; bestId[t] = -1; }

    const float* row = S + (size_t)i * n;
    for (int j = 0; j < n; ++j) {
        float v = row[j];
        if (v > bestVal[K - 1]) {                    // beats current worst
            int p = K - 1;
            while (p > 0 && bestVal[p - 1] < v) {    // shift down
                bestVal[p] = bestVal[p - 1];
                bestId [p] = bestId [p - 1];
                --p;
            }
            bestVal[p] = v;
            bestId [p] = j;
        }
    }
    #pragma unroll
    for (int t = 0; t < K; ++t) {
        out_ids[(size_t)i * K + t] = bestId [t];
        out_val[(size_t)i * K + t] = bestVal[t];
    }
}


int main()
{
    print_device_info();

    // ── Load data (host) ─────────────────────────────────────────────────────
    size_t n = 0, bd = 0, m = 0, qd = 0, gn = 0, gt_d = 0;
    float*   base  = load_data<float>  ("data/DEEP100K.base.100k.fbin",        n, bd);
    float*   query = load_data<float>  ("data/DEEP100K.query.fbin",            m, qd);
    // gt file is [int32 IDs][float32 dists]; loading as int32 reads exactly the
    // ID block (DEEP IP convention: gt distances are inner products, desc order).
    int32_t* gt    = load_data<int32_t>("data/DEEP100K.gt.query.100k.top100.bin", gn, gt_d);

    printf("[data] base n=%zu d=%zu | query m=%zu d=%zu | gt rows=%zu d=%zu\n",
           n, bd, m, qd, gn, gt_d);
    if (bd != DIM || qd != DIM) {
        fprintf(stderr, "[error] DIM mismatch (expected %d)\n", DIM);
        return 1;
    }

    // ── Device allocations ───────────────────────────────────────────────────
    float   *d_base, *d_qchunk, *d_S, *d_val;
    int32_t *d_ids;
    CUDA_CHECK(cudaMalloc(&d_base,   sizeof(float)   * n * DIM));
    CUDA_CHECK(cudaMalloc(&d_qchunk, sizeof(float)   * QCHUNK * DIM));
    CUDA_CHECK(cudaMalloc(&d_S,      sizeof(float)   * (size_t)QCHUNK * n));
    CUDA_CHECK(cudaMalloc(&d_ids,    sizeof(int32_t) * QCHUNK * K));
    CUDA_CHECK(cudaMalloc(&d_val,    sizeof(float)   * QCHUNK * K));

    std::vector<int32_t> h_ids(m * K);   // all query results gathered on host

    GpuProfile prof;
    prof.num_queries = m;
    CudaTimer timer, total;

    // ── Upload base once (counts as H2D) ─────────────────────────────────────
    total.start();
    timer.start();
    CUDA_CHECK(cudaMemcpy(d_base, base, sizeof(float) * n * DIM,
                          cudaMemcpyHostToDevice));
    prof.t_h2d_ms += timer.stop();

    // ── Process queries in chunks ────────────────────────────────────────────
    dim3 dblock(64, 4);                                    // 256 threads/block
    for (size_t q0 = 0; q0 < m; q0 += QCHUNK) {
        int qn = static_cast<int>(std::min((size_t)QCHUNK, m - q0));

        // upload this query chunk
        timer.start();
        CUDA_CHECK(cudaMemcpy(d_qchunk, query + q0 * DIM,
                              sizeof(float) * qn * DIM, cudaMemcpyHostToDevice));
        prof.t_h2d_ms += timer.stop();

        // distance GEMM (naive)
        dim3 dgrid((n + dblock.x - 1) / dblock.x,
                   (qn + dblock.y - 1) / dblock.y);
        timer.start();
        ip_distance_kernel<<<dgrid, dblock>>>(d_qchunk, d_base, d_S, qn, (int)n);
        CUDA_CHECK(cudaGetLastError());
        prof.t_compute_ms += timer.stop();

        // top-k
        int tgrid = (qn + TOPK_TPB - 1) / TOPK_TPB;
        timer.start();
        topk_kernel<<<tgrid, TOPK_TPB>>>(d_S, qn, (int)n, d_ids, d_val);
        CUDA_CHECK(cudaGetLastError());
        prof.t_topk_ms += timer.stop();

        // download results for this chunk
        timer.start();
        CUDA_CHECK(cudaMemcpy(h_ids.data() + q0 * K, d_ids,
                              sizeof(int32_t) * qn * K, cudaMemcpyDeviceToHost));
        prof.t_d2h_ms += timer.stop();
    }
    prof.t_total_ms = total.stop();

    // ── Recall + report ──────────────────────────────────────────────────────
    double recall = compute_recall(h_ids.data(), gt, gt_d, m, K);
    prof.print("GEMM-kNN brute-force (naive)");
    printf("\n[result] recall@%d = %.5f\n", K, recall);

    // ── Cleanup ──────────────────────────────────────────────────────────────
    cudaFree(d_base); cudaFree(d_qchunk); cudaFree(d_S);
    cudaFree(d_ids);  cudaFree(d_val);
    delete[] base; delete[] query; delete[] gt;
    return 0;
}
