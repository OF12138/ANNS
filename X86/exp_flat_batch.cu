// =============================================================================
// exp_flat_batch.cu -- GPU Flat (cuBLAS NNS) batch-size amortization sweep
//
// Question: how does the GPU brute-force pipeline's per-query cost change with
// batch size m?  At m=1 the fixed costs (kernel launches, PCIe transfer, GEMM
// with a degenerate 1-row matrix) cannot be amortized -- that number is the
// true SINGLE-QUERY latency of the GPU Flat path.  As m grows the fixed costs
// spread over the batch and the amortized latency (us/query) falls toward the
// throughput regime measured in exp_gemm_variants (~9.5 us/query @ m=10000).
//
// Pipeline per batch (identical math to gemm_knn.cu / exp_gemm_variants.cu,
// cuBLAS backend): H2D(query) -> S = Q.B^T via cublasSgemm -> topk_kernel ->
// D2H(ids).  Base is uploaded ONCE before the sweep (offline index-load analog)
// and is NOT counted in per-batch time; its one-time cost is printed separately.
//
// Method: for each m in the sweep, 1 warmup run + R=5 timed runs, phases are
// cudaEvent-timed and averaged over the 5 runs; min/max of total wall is also
// recorded so boost-clock noise is visible in the CSV.
//
// Build:  build.bat exp_flat_batch.cu exp_flat_batch.exe -lcublas
// Run  :  exp_flat_batch.exe          (expects ./data/DEEP100K.*)
// Out  :  results_flat_batch.csv
// =============================================================================

#include "gpu_utils.cuh"
#include <cfloat>
#include <cublas_v2.h>

#define DIM       96
#define K         10
#define QCHUNK    2048     // queries per GPU pass (S chunk = 2048 x n = 800 MB)
#define TOPK_TPB  128
#define REPEATS   5

// ── Top-k kernel: per query row, keep K largest IPs (same as gemm_knn.cu) ────
__global__ void topk_kernel(const float* __restrict__ S,
                            int qn, int n,
                            int32_t* __restrict__ out_ids,
                            float*   __restrict__ out_val)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= qn) return;

    float bestVal[K];
    int   bestId [K];
    #pragma unroll
    for (int t = 0; t < K; ++t) { bestVal[t] = -FLT_MAX; bestId[t] = -1; }

    const float* row = S + (size_t)i * n;
    for (int j = 0; j < n; ++j) {
        float v = row[j];
        if (v > bestVal[K - 1]) {
            int p = K - 1;
            while (p > 0 && bestVal[p - 1] < v) {
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

    size_t n = 0, bd = 0, mq = 0, qd = 0, gn = 0, gt_d = 0;
    float*   base  = load_data<float>  ("data/DEEP100K.base.100k.fbin",        n, bd);
    float*   query = load_data<float>  ("data/DEEP100K.query.fbin",            mq, qd);
    int32_t* gt    = load_data<int32_t>("data/DEEP100K.gt.query.100k.top100.bin", gn, gt_d);
    printf("[data] base n=%zu d=%zu | query m=%zu d=%zu | gt rows=%zu d=%zu\n",
           n, bd, mq, qd, gn, gt_d);
    if (bd != DIM || qd != DIM) {
        fprintf(stderr, "[error] DIM mismatch (expected %d)\n", DIM);
        return 1;
    }

    float   *d_base, *d_qchunk, *d_S, *d_val;
    int32_t *d_ids;
    CUDA_CHECK(cudaMalloc(&d_base,   sizeof(float)   * n * DIM));
    CUDA_CHECK(cudaMalloc(&d_qchunk, sizeof(float)   * QCHUNK * DIM));
    CUDA_CHECK(cudaMalloc(&d_S,      sizeof(float)   * (size_t)QCHUNK * n));
    CUDA_CHECK(cudaMalloc(&d_ids,    sizeof(int32_t) * QCHUNK * K));
    CUDA_CHECK(cudaMalloc(&d_val,    sizeof(float)   * QCHUNK * K));

    // one-time base upload = offline "index load", not part of per-batch time
    CudaTimer tbase;
    tbase.start();
    CUDA_CHECK(cudaMemcpy(d_base, base, sizeof(float) * n * DIM,
                          cudaMemcpyHostToDevice));
    float base_up_ms = tbase.stop();
    printf("[setup] base upload (one-time): %.3f ms  (38.4 MB)\n", base_up_ms);

    cublasHandle_t blas;
    cublasCreate(&blas);
    const float alpha = 1.0f, beta = 0.0f;

    std::vector<int32_t> h_ids(mq * K);

    // run the full pipeline for the first m queries; fills phase times (ms)
    auto run_batch = [&](int m, float& h2d, float& comp, float& topk,
                         float& d2h, float& total) {
        h2d = comp = topk = d2h = 0.0f;
        CudaTimer tp, tt;
        tt.start();
        for (int q0 = 0; q0 < m; q0 += QCHUNK) {
            int qn = std::min(QCHUNK, m - q0);

            tp.start();
            CUDA_CHECK(cudaMemcpy(d_qchunk, query + (size_t)q0 * DIM,
                                  sizeof(float) * qn * DIM,
                                  cudaMemcpyHostToDevice));
            h2d += tp.stop();

            // row-major S(qn x n) == col-major C(n x qn) = B^T(n x d) . Q(d x qn)
            tp.start();
            cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N,
                        (int)n, qn, DIM,
                        &alpha, d_base, DIM, d_qchunk, DIM,
                        &beta,  d_S, (int)n);
            comp += tp.stop();

            tp.start();
            topk_kernel<<<(qn + TOPK_TPB - 1) / TOPK_TPB, TOPK_TPB>>>(
                d_S, qn, (int)n, d_ids, d_val);
            CUDA_CHECK(cudaGetLastError());
            topk += tp.stop();

            tp.start();
            CUDA_CHECK(cudaMemcpy(h_ids.data() + (size_t)q0 * K, d_ids,
                                  sizeof(int32_t) * qn * K,
                                  cudaMemcpyDeviceToHost));
            d2h += tp.stop();
        }
        total = tt.stop();
    };

    // cuBLAS context warmup so the first timed m=1 run isn't paying init cost
    {
        float a, b, c, d, e;
        run_batch(64, a, b, c, d, e);
    }

    const int sweep[] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                          1024, 2048, 4096, 8192, 10000 };
    const int nsweep = sizeof(sweep) / sizeof(sweep[0]);

    FILE* csv = fopen("results_flat_batch.csv", "w");
    fprintf(csv, "m,h2d_ms,compute_ms,topk_ms,d2h_ms,total_ms,"
                 "total_min_ms,total_max_ms,us_per_query,qps,recall,runs\n");
    printf("\n%7s %10s %10s %10s %10s %10s %12s %10s\n",
           "m", "h2d_ms", "comp_ms", "topk_ms", "d2h_ms", "total_ms",
           "us/query", "recall");

    for (int s = 0; s < nsweep; ++s) {
        int m = sweep[s];
        if ((size_t)m > mq) break;

        float h2d, comp, topk, d2h, total;
        run_batch(m, h2d, comp, topk, d2h, total);          // warmup for this m

        float ah = 0, ac = 0, ak = 0, ad = 0, at = 0;
        float tmin = FLT_MAX, tmax = 0;
        for (int r = 0; r < REPEATS; ++r) {
            run_batch(m, h2d, comp, topk, d2h, total);
            ah += h2d; ac += comp; ak += topk; ad += d2h; at += total;
            tmin = std::min(tmin, total);
            tmax = std::max(tmax, total);
        }
        ah /= REPEATS; ac /= REPEATS; ak /= REPEATS; ad /= REPEATS; at /= REPEATS;

        double recall = compute_recall(h_ids.data(), gt, gt_d, m, K);
        double uspq   = 1000.0 * at / m;
        double qps    = m / (at / 1000.0);

        printf("%7d %10.3f %10.3f %10.3f %10.3f %10.3f %12.3f %10.5f\n",
               m, ah, ac, ak, ad, at, uspq, recall);
        fprintf(csv, "%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.1f,%.5f,%d\n",
                m, ah, ac, ak, ad, at, tmin, tmax, uspq, qps, recall, REPEATS);
    }
    fclose(csv);
    printf("\n[done] raw data written to results_flat_batch.csv\n");

    cublasDestroy(blas);
    cudaFree(d_base); cudaFree(d_qchunk); cudaFree(d_S);
    cudaFree(d_ids);  cudaFree(d_val);
    delete[] base; delete[] query; delete[] gt;
    return 0;
}
