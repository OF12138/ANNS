// =============================================================================
// exp_gemm_variants.cu -- latency comparison of three GEMM-kNN distance backends
//
//   (1) BASELINE  ip_distance_kernel        -- naive: 1 thread per (query,base),
//                                               no reuse  (same as gemm_knn.cu)
//   (2) TILING    ip_distance_kernel_tiled  -- shared-memory tiled GEMM
//   (3) cuBLAS    cublasSgemm               -- vendor BLAS
//
// All three produce the SAME row-major score matrix S[i*n+j] = IP(query_i,base_j)
// and then run the SAME topk_kernel, so only the distance backend differs.
// We compare LATENCY: distance-compute time, top-k time, total pipeline time,
// and amortized us/query.  Recall is printed only to confirm correctness.
//
//   >>> CORE CODE FOR THE REPORT <<<
//     TILING : ip_distance_kernel_tiled()  (this file, search "TILING CORE")
//     cuBLAS : run_distance(), MODE_CUBLAS branch (this file, "CUBLAS CORE")
//
// Build:  build.bat exp_gemm_variants.cu exp_gemm_variants.exe -lcublas
// Run  :  ./exp_gemm_variants
// =============================================================================

#include "gpu_utils.cuh"
#include <cfloat>
#include <cublas_v2.h>

#define DIM       96
#define K         10
#define QCHUNK    2048
#define TOPK_TPB  128
#define TILE      32        // tile edge for the tiled GEMM (block = TILE x TILE)


// ── (1) BASELINE distance kernel (naive, identical to gemm_knn.cu) ───────────
__global__ void ip_distance_kernel(const float* __restrict__ Q,
                                   const float* __restrict__ B,
                                   float* __restrict__ S, int qn, int n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;   // base
    int i = blockIdx.y * blockDim.y + threadIdx.y;   // query
    if (i >= qn || j >= n) return;
    const float* q = Q + (size_t)i * DIM;
    const float* b = B + (size_t)j * DIM;
    float acc = 0.0f;
    #pragma unroll
    for (int t = 0; t < DIM; ++t) acc += q[t] * b[t];
    S[(size_t)i * n + j] = acc;
}


// ── (2) TILING CORE: shared-memory tiled GEMM  S = Q · Bᵀ ─────────────────────
// Each block computes a TILE x TILE tile of S.  Q and B tiles are staged in
// shared memory and reused by all TILE threads along a row/column, cutting
// global-memory traffic by ~TILE x versus the naive kernel.
// sQ[ty][t] = Q[row][t0+t]   ;  sB[t][tx] = B[col][t0+t]
// acc += sQ[ty][t] * sB[t][tx]  accumulates the inner product over d in TILE steps.
__global__ void ip_distance_kernel_tiled(const float* __restrict__ Q,
                                         const float* __restrict__ B,
                                         float* __restrict__ S, int qn, int n)
{
    __shared__ float sQ[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;   // query index
    int col = blockIdx.x * TILE + threadIdx.x;   // base index
    float acc = 0.0f;

    for (int t0 = 0; t0 < DIM; t0 += TILE) {
        // stage Q tile: thread(tx,ty) loads Q[row][t0+tx]  (coalesced over tx)
        sQ[threadIdx.y][threadIdx.x] =
            (row < qn && (t0 + threadIdx.x) < DIM)
            ? Q[(size_t)row * DIM + t0 + threadIdx.x] : 0.0f;
        // stage B tile: thread(tx,ty) loads B[col][t0+ty]
        sB[threadIdx.y][threadIdx.x] =
            (col < n && (t0 + threadIdx.y) < DIM)
            ? B[(size_t)col * DIM + t0 + threadIdx.y] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int t = 0; t < TILE; ++t)
            acc += sQ[threadIdx.y][t] * sB[t][threadIdx.x];
        __syncthreads();
    }
    if (row < qn && col < n) S[(size_t)row * n + col] = acc;
}


// ── shared top-k kernel (one thread per query) ───────────────────────────────
__global__ void topk_kernel(const float* __restrict__ S, int qn, int n,
                            int32_t* __restrict__ out_ids,
                            float* __restrict__ out_val)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= qn) return;
    float bestVal[K]; int bestId[K];
    #pragma unroll
    for (int t = 0; t < K; ++t) { bestVal[t] = -FLT_MAX; bestId[t] = -1; }
    const float* row = S + (size_t)i * n;
    for (int j = 0; j < n; ++j) {
        float v = row[j];
        if (v > bestVal[K - 1]) {
            int p = K - 1;
            while (p > 0 && bestVal[p - 1] < v) {
                bestVal[p] = bestVal[p - 1]; bestId[p] = bestId[p - 1]; --p;
            }
            bestVal[p] = v; bestId[p] = j;
        }
    }
    #pragma unroll
    for (int t = 0; t < K; ++t) {
        out_ids[(size_t)i * K + t] = bestId[t];
        out_val[(size_t)i * K + t] = bestVal[t];
    }
}


enum Mode { MODE_BASELINE = 0, MODE_TILED = 1, MODE_CUBLAS = 2 };
static const char* MODE_NAME[] = { "baseline-naive", "tiling-shared", "cuBLAS" };


// dispatch one chunk's distance computation into d_S (qn x n, row-major)
static void run_distance(int mode, cublasHandle_t blas,
                         const float* d_Q, const float* d_B, float* d_S,
                         int qn, int n)
{
    if (mode == MODE_BASELINE) {
        dim3 blk(64, 4);
        dim3 grd((n + blk.x - 1) / blk.x, (qn + blk.y - 1) / blk.y);
        ip_distance_kernel<<<grd, blk>>>(d_Q, d_B, d_S, qn, n);
    } else if (mode == MODE_TILED) {
        dim3 blk(TILE, TILE);
        dim3 grd((n + TILE - 1) / TILE, (qn + TILE - 1) / TILE);
        ip_distance_kernel_tiled<<<grd, blk>>>(d_Q, d_B, d_S, qn, n);
    } else { // ── CUBLAS CORE ──────────────────────────────────────────────
        // Want row-major S(qn x n), S[i*n+j] = sum_t Q[i][t]*B[j][t].
        // Row-major S(qn x n) == col-major C(n x qn).  In column-major:
        //   C(n x qn) = op(Bᵀ stored d x n)·op(Q stored d x qn), k = d.
        // cublasSgemm(handle, OP_T, OP_N, n, qn, d, a, dB, d, dQ, d, b, dS, n)
        const float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N,
                    n, qn, DIM,
                    &alpha, d_B, DIM, d_Q, DIM,
                    &beta,  d_S, n);
    }
}


int main()
{
    print_device_info();

    size_t n, bd, m, qd, gn, gt_d;
    float*   base  = load_data<float>  ("data/DEEP100K.base.100k.fbin",         n, bd);
    float*   query = load_data<float>  ("data/DEEP100K.query.fbin",             m, qd);
    int32_t* gt    = load_data<int32_t>("data/DEEP100K.gt.query.100k.top100.bin", gn, gt_d);
    printf("[data] base n=%zu d=%zu | query m=%zu | gt d=%zu\n", n, bd, m, gt_d);

    float   *d_base, *d_qchunk, *d_S, *d_val;
    int32_t *d_ids;
    CUDA_CHECK(cudaMalloc(&d_base,   sizeof(float)   * n * DIM));
    CUDA_CHECK(cudaMalloc(&d_qchunk, sizeof(float)   * QCHUNK * DIM));
    CUDA_CHECK(cudaMalloc(&d_S,      sizeof(float)   * (size_t)QCHUNK * n));
    CUDA_CHECK(cudaMalloc(&d_ids,    sizeof(int32_t) * QCHUNK * K));
    CUDA_CHECK(cudaMalloc(&d_val,    sizeof(float)   * QCHUNK * K));
    CUDA_CHECK(cudaMemcpy(d_base, base, sizeof(float) * n * DIM, cudaMemcpyHostToDevice));

    cublasHandle_t blas; cublasCreate(&blas);
    std::vector<int32_t> ids(m * K);

    // run the full m-query pipeline for one mode; returns timings (ms) by ref
    auto run_mode = [&](int mode, float& compute_ms, float& topk_ms,
                        float& total_ms) {
        CudaTimer tc, tk, tt;
        compute_ms = topk_ms = 0.0f;
        tt.start();
        for (size_t q0 = 0; q0 < m; q0 += QCHUNK) {
            int qn = (int)std::min((size_t)QCHUNK, m - q0);
            CUDA_CHECK(cudaMemcpy(d_qchunk, query + q0 * DIM,
                                  sizeof(float) * qn * DIM, cudaMemcpyHostToDevice));
            tc.start();
            run_distance(mode, blas, d_qchunk, d_base, d_S, qn, (int)n);
            compute_ms += tc.stop();

            int tgrid = (qn + TOPK_TPB - 1) / TOPK_TPB;
            tk.start();
            topk_kernel<<<tgrid, TOPK_TPB>>>(d_S, qn, (int)n, d_ids, d_val);
            topk_ms += tk.stop();

            CUDA_CHECK(cudaMemcpy(ids.data() + q0 * K, d_ids,
                                  sizeof(int32_t) * qn * K, cudaMemcpyDeviceToHost));
        }
        total_ms = tt.stop();
    };

    const int REP = 5;
    const char* header =
        "variant,compute_ms,topk_ms,total_ms,us_per_query,compute_speedup,recall";
    FILE* csv = fopen("results_gemm_variants.csv", "w");
    printf("\n%s\n", header);
    fprintf(csv, "%s\n", header);

    float base_compute = 0.0f;
    for (int mode = 0; mode < 3; ++mode) {
        // best (min) over REP reps to suppress boost-clock jitter
        float bc = 1e30f, bt = 1e30f, btot = 1e30f;
        for (int r = 0; r < REP; ++r) {
            float c, t, tot; run_mode(mode, c, t, tot);
            if (tot < btot) { btot = tot; bc = c; bt = t; }
        }
        double recall = compute_recall(ids.data(), gt, gt_d, m, K);
        if (mode == MODE_BASELINE) base_compute = bc;

        char line[512];
        snprintf(line, sizeof(line), "%s,%.3f,%.3f,%.3f,%.3f,%.2f,%.5f",
                 MODE_NAME[mode], bc, bt, btot,
                 1000.0 * btot / m, base_compute / bc, recall);
        printf("%s\n", line);
        fprintf(csv, "%s\n", line);
    }
    fclose(csv);
    printf("\n[done] raw data written to results_gemm_variants.csv\n");

    cublasDestroy(blas);
    cudaFree(d_base); cudaFree(d_qchunk); cudaFree(d_S);
    cudaFree(d_ids);  cudaFree(d_val);
    delete[] base; delete[] query; delete[] gt;
    return 0;
}
