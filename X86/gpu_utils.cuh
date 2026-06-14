// =============================================================================
// gpu_utils.cuh -- shared utilities + unified profiling for all GPU ANNS programs
//
// Every GPU program in this experiment includes this header so that:
//   1. Data loading is identical (DEEP100K .fbin / .bin format).
//   2. Profiling is reported in the SAME phase breakdown, making the matrix-mul
//      baseline, its optimized variants, and the IVF programs directly
//      comparable in the report.
//
// Profiling model (GpuProfile):
//   t_h2d_ms     host->device transfer (base + query upload)
//   t_compute_ms distance computation kernel(s)  -- the core GPU work
//   t_topk_ms    top-k selection kernel(s)
//   t_d2h_ms     device->host transfer (results back)
//   t_total_ms   full end-to-end wall time (includes the above + launch gaps)
//   Derived:     throughput = num_queries / (t_total_ms/1000)  [queries/sec]
//
// Timing method:
//   GPU phases use cudaEvent (≈0.5 us resolution, measures real device time).
//   t_total uses a cudaEvent pair around the whole pipeline.
//   All kernel-phase timers accumulate across query chunks.
// =============================================================================
#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <set>
#include <algorithm>


// ── CUDA error checking ──────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)


// ── Data loading (DEEP100K format: int32 n, int32 d, then n*d payload) ───────
template<typename T>
static T* load_data(const std::string& path, size_t& n, size_t& d)
{
    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "[load] cannot open %s\n", path.c_str());
        exit(EXIT_FAILURE);
    }
    uint32_t hn = 0, hd = 0;
    fin.read(reinterpret_cast<char*>(&hn), 4);
    fin.read(reinterpret_cast<char*>(&hd), 4);
    n = hn; d = hd;
    T* data = new T[n * d];
    fin.read(reinterpret_cast<char*>(data), sizeof(T) * n * d);
    fin.close();
    return data;
}


// ── cudaEvent timer wrapper ──────────────────────────────────────────────────
struct CudaTimer {
    cudaEvent_t beg, end;
    CudaTimer()  { cudaEventCreate(&beg); cudaEventCreate(&end); }
    ~CudaTimer() { cudaEventDestroy(beg); cudaEventDestroy(end); }
    void start() { cudaEventRecord(beg, 0); }
    // returns elapsed milliseconds since start()
    float stop() {
        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, beg, end);
        return ms;
    }
};


// ── Unified profiling record ─────────────────────────────────────────────────
struct GpuProfile {
    float  t_h2d_ms     = 0.0f;
    float  t_compute_ms = 0.0f;
    float  t_topk_ms    = 0.0f;
    float  t_d2h_ms     = 0.0f;
    float  t_total_ms   = 0.0f;
    size_t num_queries  = 0;

    void print(const char* tag) const {
        double tot = t_total_ms > 0 ? t_total_ms : 1e-9;
        double qps = num_queries / (tot / 1000.0);
        printf("\n==================== PROFILE: %s ====================\n", tag);
        printf("  queries            : %zu\n", num_queries);
        printf("  ---- phase breakdown (device time) ----\n");
        printf("  H2D  transfer      : %10.3f ms  (%5.1f%%)\n",
               t_h2d_ms,     100.0 * t_h2d_ms     / tot);
        printf("  compute (distance) : %10.3f ms  (%5.1f%%)\n",
               t_compute_ms, 100.0 * t_compute_ms / tot);
        printf("  top-k selection    : %10.3f ms  (%5.1f%%)\n",
               t_topk_ms,    100.0 * t_topk_ms    / tot);
        printf("  D2H  transfer      : %10.3f ms  (%5.1f%%)\n",
               t_d2h_ms,     100.0 * t_d2h_ms     / tot);
        printf("  ---------------------------------------\n");
        printf("  TOTAL  (wall)      : %10.3f ms\n", t_total_ms);
        printf("  throughput         : %10.1f queries/sec\n", qps);
        printf("  amortized latency  : %10.3f us/query\n",
               1000.0 * tot / num_queries);
        printf("=========================================================\n");
    }
};


// ── Recall@k against int32 ground-truth (top-100 per query) ──────────────────
// gt file layout: header(n,d) then [n*d int32 IDs][n*d float32 distances].
// load_data<int32_t> reads exactly the int32 ID block (stops before distances),
// so `gt` here is [num_queries x gt_d] int32, row i = sorted true neighbor IDs.
// our_ids: [num_queries x k] int32, row i = our predicted neighbor IDs.
// recall@k = mean over queries of |our_top_k ∩ gt_top_k| / k.
static double compute_recall(const int32_t* our_ids,
                             const int32_t* gt, size_t gt_d,
                             size_t num_queries, size_t k)
{
    double acc = 0.0;
    for (size_t i = 0; i < num_queries; ++i) {
        std::set<int32_t> truth;
        for (size_t j = 0; j < k; ++j)
            truth.insert(gt[i * gt_d + j]);
        size_t hit = 0;
        for (size_t j = 0; j < k; ++j)
            if (truth.count(our_ids[i * k + j])) ++hit;
        acc += static_cast<double>(hit) / k;
    }
    return acc / num_queries;
}


// ── One-shot device info print (called once at program start) ────────────────
static void print_device_info()
{
    int dev = 0;
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    printf("[device] %s  | SM %d.%d | %d SMs | %.1f GB VRAM | %.0f MHz\n",
           p.name, p.major, p.minor, p.multiProcessorCount,
           p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
           p.clockRate / 1000.0);
}
