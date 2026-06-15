// =============================================================================
// exp_cpu_gpu.cu -- CPU vs GPU brute-force kNN contrast experiment
//
// Runs the SAME exact brute-force kNN (inner-product, top-k largest) three ways
// on identical data, sweeping the query-batch size m:
//   1. CPU single-thread        (scalar, auto-vectorized by /O2)
//   2. CPU multi-thread (OpenMP, all logical cores)
//   3. GPU GEMM kNN     (naive ip_distance_kernel + topk_kernel, same as
//                        gemm_knn.cu baseline)
//
// For each m it prints one CSV row so the numbers can be charted in Excel.
// Timing convention (fair contrast, data already in RAM / VRAM):
//   - GPU: base resident in VRAM (uploaded once outside the loop). Per-batch
//          time = query H2D + distance kernel + top-k kernel + result D2H.
//   - CPU: wall time of the brute-force loop only.
//   Disk loading is excluded from all timings.
//
// Build:  build.bat exp_cpu_gpu.cu exp_cpu_gpu.exe -Xcompiler /openmp
// Run  :  ./exp_cpu_gpu   (writes results_cpu_gpu.csv and echoes to stdout)
// =============================================================================

#include "gpu_utils.cuh"
#include <cfloat>
#include <chrono>
#include <omp.h>

#define DIM       96
#define K         10
#define QCHUNK    2048
#define TOPK_TPB  128


// ── GPU kernels (identical to gemm_knn.cu baseline) ──────────────────────────
__global__ void ip_distance_kernel(const float* __restrict__ Q,
                                    const float* __restrict__ B,
                                    float* __restrict__ S, int qn, int n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= qn || j >= n) return;
    const float* q = Q + (size_t)i * DIM;
    const float* b = B + (size_t)j * DIM;
    float acc = 0.0f;
    #pragma unroll
    for (int t = 0; t < DIM; ++t) acc += q[t] * b[t];
    S[(size_t)i * n + j] = acc;
}

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


// ── CPU brute-force kNN (inner product, top-k largest), nthreads workers ─────
static void cpu_knn(const float* base, const float* query,
                    size_t m, size_t n, int32_t* out_ids, int nthreads)
{
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (long long i = 0; i < (long long)m; ++i) {
        float bestVal[K]; int bestId[K];
        for (int t = 0; t < K; ++t) { bestVal[t] = -FLT_MAX; bestId[t] = -1; }
        const float* q = query + (size_t)i * DIM;
        for (size_t j = 0; j < n; ++j) {
            const float* b = base + j * DIM;
            float acc = 0.0f;
            for (int t = 0; t < DIM; ++t) acc += q[t] * b[t];
            if (acc > bestVal[K - 1]) {
                int p = K - 1;
                while (p > 0 && bestVal[p - 1] < acc) {
                    bestVal[p] = bestVal[p - 1]; bestId[p] = bestId[p - 1]; --p;
                }
                bestVal[p] = acc; bestId[p] = (int)j;
            }
        }
        for (int t = 0; t < K; ++t) out_ids[i * K + t] = bestId[t];
    }
}


int main()
{
    print_device_info();
    const int maxthreads = omp_get_max_threads();
    printf("[cpu] OpenMP max threads = %d\n", maxthreads);

    // ── Load data ────────────────────────────────────────────────────────────
    size_t n, bd, m_all, qd, gn, gt_d;
    float*   base  = load_data<float>  ("data/DEEP100K.base.100k.fbin",         n, bd);
    float*   query = load_data<float>  ("data/DEEP100K.query.fbin",             m_all, qd);
    int32_t* gt    = load_data<int32_t>("data/DEEP100K.gt.query.100k.top100.bin", gn, gt_d);
    printf("[data] base n=%zu d=%zu | query m=%zu | gt d=%zu\n", n, bd, m_all, gt_d);

    // ── Device buffers (base resident; chunk buffers sized for QCHUNK) ────────
    float   *d_base, *d_qchunk, *d_S, *d_val;
    int32_t *d_ids;
    CUDA_CHECK(cudaMalloc(&d_base,   sizeof(float)   * n * DIM));
    CUDA_CHECK(cudaMalloc(&d_qchunk, sizeof(float)   * QCHUNK * DIM));
    CUDA_CHECK(cudaMalloc(&d_S,      sizeof(float)   * (size_t)QCHUNK * n));
    CUDA_CHECK(cudaMalloc(&d_ids,    sizeof(int32_t) * QCHUNK * K));
    CUDA_CHECK(cudaMalloc(&d_val,    sizeof(float)   * QCHUNK * K));
    CUDA_CHECK(cudaMemcpy(d_base, base, sizeof(float) * n * DIM, cudaMemcpyHostToDevice));

    std::vector<int32_t> ids_gpu(m_all * K), ids_cpu(m_all * K);

    // ── GPU batch runner over the first m queries (base already resident) ────
    auto gpu_run = [&](size_t m) -> float {
        CudaTimer total; total.start();
        dim3 dblock(64, 4);
        for (size_t q0 = 0; q0 < m; q0 += QCHUNK) {
            int qn = (int)std::min((size_t)QCHUNK, m - q0);
            CUDA_CHECK(cudaMemcpy(d_qchunk, query + q0 * DIM,
                                  sizeof(float) * qn * DIM, cudaMemcpyHostToDevice));
            dim3 dgrid((n + dblock.x - 1) / dblock.x, (qn + dblock.y - 1) / dblock.y);
            ip_distance_kernel<<<dgrid, dblock>>>(d_qchunk, d_base, d_S, qn, (int)n);
            int tgrid = (qn + TOPK_TPB - 1) / TOPK_TPB;
            topk_kernel<<<tgrid, TOPK_TPB>>>(d_S, qn, (int)n, d_ids, d_val);
            CUDA_CHECK(cudaMemcpy(ids_gpu.data() + q0 * K, d_ids,
                                  sizeof(int32_t) * qn * K, cudaMemcpyDeviceToHost));
        }
        return total.stop();
    };

    // ── Sweep ────────────────────────────────────────────────────────────────
    const size_t sweep[] = {100, 250, 500, 1000, 2000, 4000, 6000, 8000, 10000};
    const int    ns = sizeof(sweep) / sizeof(sweep[0]);

    FILE* csv = fopen("results_cpu_gpu.csv", "w");
    const char* header =
        "m,cpu1_ms,cpuN_ms,gpu_ms,cpu1_qps,cpuN_qps,gpu_qps,"
        "speedup_gpu_vs_cpu1,speedup_gpu_vs_cpuN,speedup_cpuN_vs_cpu1,recall";
    printf("\n%s\n", header);
    fprintf(csv, "%s\n", header);

    // warm-up GPU (kernel JIT / clocks) so first timed row isn't penalized
    gpu_run(256);

    // Report the BEST (min) time over repeats per config to suppress noise.
    // Phases are DECOUPLED: all GPU runs first (GPU cold, full boost clocks),
    // then CPU. On this shared-cooling laptop, running a 30 s CPU burn before a
    // GPU timing throttles GPU boost and corrupts the contrast, so we isolate.
    const int REP_CPUN = 5, REP_GPU = 5;
    std::vector<double> v_gpu(ns), v_cpuN(ns), v_cpu1(ns), v_recall(ns);

    // ---- Phase 1: GPU (best of REP_GPU) ----
    for (int s = 0; s < ns; ++s) {
        double best = 1e30;
        for (int r = 0; r < REP_GPU; ++r) best = std::min(best, gpu_run(sweep[s]));
        v_gpu[s]    = best;
        v_recall[s] = compute_recall(ids_gpu.data(), gt, gt_d, sweep[s], K);
    }
    // ---- Phase 2: CPU multi-thread (best of REP_CPUN) ----
    for (int s = 0; s < ns; ++s) {
        double best = 1e30;
        for (int r = 0; r < REP_CPUN; ++r) {
            auto d0 = std::chrono::high_resolution_clock::now();
            cpu_knn(base, query, sweep[s], n, ids_cpu.data(), maxthreads);
            auto d1 = std::chrono::high_resolution_clock::now();
            best = std::min(best,
                std::chrono::duration<double, std::milli>(d1 - d0).count());
        }
        v_cpuN[s] = best;
    }
    // ---- Phase 3: CPU single-thread (1 run; stable) ----
    for (int s = 0; s < ns; ++s) {
        auto c0 = std::chrono::high_resolution_clock::now();
        cpu_knn(base, query, sweep[s], n, ids_cpu.data(), 1);
        auto c1 = std::chrono::high_resolution_clock::now();
        v_cpu1[s] = std::chrono::duration<double, std::milli>(c1 - c0).count();
    }

    // ---- Emit table ----
    for (int s = 0; s < ns; ++s) {
        size_t m = sweep[s];
        double cpu1_ms = v_cpu1[s], cpuN_ms = v_cpuN[s], gpu_ms = v_gpu[s];
        char line[512];
        snprintf(line, sizeof(line),
            "%zu,%.3f,%.3f,%.3f,%.1f,%.1f,%.1f,%.2f,%.2f,%.2f,%.5f",
            m, cpu1_ms, cpuN_ms, gpu_ms,
            m / (cpu1_ms / 1000.0), m / (cpuN_ms / 1000.0), m / (gpu_ms / 1000.0),
            cpu1_ms / gpu_ms, cpuN_ms / gpu_ms, cpu1_ms / cpuN_ms, v_recall[s]);
        printf("%s\n", line);
        fprintf(csv, "%s\n", line);
    }
    fclose(csv);
    printf("\n[done] raw data written to results_cpu_gpu.csv\n");

    cudaFree(d_base); cudaFree(d_qchunk); cudaFree(d_S);
    cudaFree(d_ids);  cudaFree(d_val);
    delete[] base; delete[] query; delete[] gt;
    return 0;
}
