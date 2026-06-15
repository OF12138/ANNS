// =============================================================================
// exp_ivf_opt.cu -- optimized IVF fine scan (warp-per-query) vs original
//
// ORIGINAL  ivf_fine_kernel       : one THREAD per query.  Serial 96-dim dot
//                                    product; adjacent threads touch unrelated
//                                    cluster regions -> uncoalesced loads; only
//                                    m threads -> low occupancy.
// OPTIMIZED ivf_fine_kernel_warp   : one WARP (32 lanes) per query.  Lane l
//                                    handles dims l, l+32, l+64 (96 = 3*32),
//                                    warp-shuffle reduces the inner product.
//                                    -> coalesced 128 B loads, parallel dot,
//                                       32x more threads.  Same algorithm, so
//                                       recall is identical.
//
// Sweeps nlist x nprobe, runs BOTH fine kernels (coarse GEMM + cluster select
// are shared/identical), and reports fine-phase and total-pipeline latency plus
// speedup, so the win is attributable to the fine kernel.
//
//   >>> OPTIMIZATION CORE (for the report): ivf_fine_kernel_warp(), "OPT CORE" <<<
//
// Build:  build.bat exp_ivf_opt.cu exp_ivf_opt.exe -Xcompiler /openmp -lcublas
// Run  :  ./exp_ivf_opt
// =============================================================================

#include "gpu_utils.cuh"
#include <cfloat>
#include <cmath>
#include <random>
#include <cublas_v2.h>
#include <omp.h>

#define DIM        96
#define K          10
#define QCHUNK     2048
#define TPB        128
#define KM_ITERS   15
#define MAXNPROBE  128
#define MAX_NLIST  4096

static const int NLIST_SWEEP[]  = {256, 1024, 4096};
static const int NPROBE_SWEEP[] = {1, 2, 4, 8, 16, 32, 64, 128};


// ── IVF index + build + cache (same as exp_ivf_pareto.cu) ────────────────────
struct IVFIndex {
    int nlist = 0, d = DIM, n = 0;
    std::vector<float> centroids, reordered_base;
    std::vector<int>   cluster_off, id_map;
};

static void build_ivf(IVFIndex& idx, const float* base, int n, int d, int nlist)
{
    idx.n = n; idx.d = d; idx.nlist = nlist;
    idx.centroids.assign((size_t)nlist * d, 0.0f);
    std::vector<int> assign(n, 0);
    std::mt19937 rng(42);
    std::vector<int> perm(n);
    for (int i = 0; i < n; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    for (int c = 0; c < nlist; ++c)
        std::copy(base + (size_t)perm[c] * d, base + (size_t)perm[c] * d + d,
                  idx.centroids.begin() + (size_t)c * d);
    auto assign_step = [&]() {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            const float* x = base + (size_t)i * d;
            float best = -FLT_MAX; int bc = 0;
            for (int c = 0; c < nlist; ++c) {
                const float* ce = idx.centroids.data() + (size_t)c * d;
                float ip = 0.0f;
                for (int t = 0; t < d; ++t) ip += x[t] * ce[t];
                if (ip > best) { best = ip; bc = c; }
            }
            assign[i] = bc;
        }
    };
    for (int it = 0; it < KM_ITERS; ++it) {
        assign_step();
        std::vector<double> sum((size_t)nlist * d, 0.0);
        std::vector<int>    cnt(nlist, 0);
        for (int i = 0; i < n; ++i) {
            const float* x = base + (size_t)i * d;
            double* s = sum.data() + (size_t)assign[i] * d;
            for (int t = 0; t < d; ++t) s[t] += x[t];
            cnt[assign[i]]++;
        }
        for (int c = 0; c < nlist; ++c) {
            float* ce = idx.centroids.data() + (size_t)c * d;
            if (cnt[c] == 0) { int r = perm[(c + it) % n];
                std::copy(base + (size_t)r * d, base + (size_t)r * d + d, ce); continue; }
            double nrm = 0.0;
            for (int t = 0; t < d; ++t) { double v = sum[(size_t)c * d + t] / cnt[c]; ce[t] = (float)v; nrm += v * v; }
            nrm = std::sqrt(nrm);
            if (nrm > 1e-12) for (int t = 0; t < d; ++t) ce[t] /= (float)nrm;
        }
    }
    assign_step();
    idx.cluster_off.assign(nlist + 1, 0);
    for (int i = 0; i < n; ++i) idx.cluster_off[assign[i] + 1]++;
    for (int c = 0; c < nlist; ++c) idx.cluster_off[c + 1] += idx.cluster_off[c];
    idx.reordered_base.assign((size_t)n * d, 0.0f);
    idx.id_map.assign(n, 0);
    std::vector<int> cursor(idx.cluster_off.begin(), idx.cluster_off.end());
    for (int i = 0; i < n; ++i) {
        int pos = cursor[assign[i]]++;
        std::copy(base + (size_t)i * d, base + (size_t)i * d + d,
                  idx.reordered_base.begin() + (size_t)pos * d);
        idx.id_map[pos] = i;
    }
}
static bool load_ivf_cache(IVFIndex& idx, const char* path, const float* base, int want_nlist)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;
    int nlist, d, n;
    f.read((char*)&nlist, 4); f.read((char*)&d, 4); f.read((char*)&n, 4);
    if (nlist != want_nlist || d != DIM) return false;
    idx.nlist = nlist; idx.d = d; idx.n = n;
    idx.centroids.resize((size_t)nlist * d);
    idx.cluster_off.resize(nlist + 1);
    idx.id_map.resize(n);
    f.read((char*)idx.centroids.data(), sizeof(float) * (size_t)nlist * d);
    f.read((char*)idx.cluster_off.data(), sizeof(int) * (nlist + 1));
    f.read((char*)idx.id_map.data(), sizeof(int) * n);
    if (!f) return false;
    idx.reordered_base.resize((size_t)n * d);
    for (int pos = 0; pos < n; ++pos)
        std::copy(base + (size_t)idx.id_map[pos] * d, base + (size_t)idx.id_map[pos] * d + d,
                  idx.reordered_base.begin() + (size_t)pos * d);
    return true;
}
static void save_ivf_cache(const IVFIndex& idx, const char* path)
{
    std::ofstream f(path, std::ios::binary);
    f.write((char*)&idx.nlist, 4); f.write((char*)&idx.d, 4); f.write((char*)&idx.n, 4);
    f.write((char*)idx.centroids.data(), sizeof(float) * idx.centroids.size());
    f.write((char*)idx.cluster_off.data(), sizeof(int) * idx.cluster_off.size());
    f.write((char*)idx.id_map.data(), sizeof(int) * idx.id_map.size());
}


// ── coarse cluster selection (shared by both fine kernels) ───────────────────
__global__ void select_topn_kernel(const float* __restrict__ S, int qn, int nlist,
                                   int nprobe, int* __restrict__ cids)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= qn) return;
    float bv[MAXNPROBE]; int bi[MAXNPROBE];
    for (int p = 0; p < nprobe; ++p) { bv[p] = -FLT_MAX; bi[p] = 0; }
    const float* row = S + (size_t)i * nlist;
    for (int c = 0; c < nlist; ++c) {
        float v = row[c];
        if (v > bv[nprobe - 1]) {
            int p = nprobe - 1;
            while (p > 0 && bv[p - 1] < v) { bv[p] = bv[p - 1]; bi[p] = bi[p - 1]; --p; }
            bv[p] = v; bi[p] = c;
        }
    }
    for (int p = 0; p < nprobe; ++p) cids[(size_t)i * nprobe + p] = bi[p];
}


// ── ORIGINAL fine kernel: one thread per query ───────────────────────────────
__global__ void ivf_fine_kernel(const float* __restrict__ Q, const float* __restrict__ RB,
                                const int* __restrict__ coff, const int* __restrict__ idmap,
                                const int* __restrict__ cids, int qn, int nprobe,
                                int32_t* __restrict__ out_ids, float* __restrict__ out_val)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= qn) return;
    float bestVal[K]; int bestId[K];
    #pragma unroll
    for (int t = 0; t < K; ++t) { bestVal[t] = -FLT_MAX; bestId[t] = -1; }
    const float* q = Q + (size_t)i * DIM;
    for (int p = 0; p < nprobe; ++p) {
        int c = cids[(size_t)i * nprobe + p];
        int lo = coff[c], hi = coff[c + 1];
        for (int pos = lo; pos < hi; ++pos) {
            const float* b = RB + (size_t)pos * DIM;
            float acc = 0.0f;
            #pragma unroll
            for (int t = 0; t < DIM; ++t) acc += q[t] * b[t];
            if (acc > bestVal[K - 1]) {
                int pp = K - 1;
                while (pp > 0 && bestVal[pp - 1] < acc) { bestVal[pp] = bestVal[pp - 1]; bestId[pp] = bestId[pp - 1]; --pp; }
                bestVal[pp] = acc; bestId[pp] = idmap[pos];
            }
        }
    }
    #pragma unroll
    for (int t = 0; t < K; ++t) { out_ids[(size_t)i * K + t] = bestId[t]; out_val[(size_t)i * K + t] = bestVal[t]; }
}


// ── OPT CORE: warp-per-query fine kernel ─────────────────────────────────────
// 32 lanes cooperate on one query.  Each candidate's IP is computed by all 32
// lanes (lane l: dims l, l+32, l+64) and reduced with __shfl_down_sync; lane 0
// keeps the top-k.  Loads of RB are coalesced (32 consecutive floats per warp
// step), the dot product is parallel, and there are 32x more threads in flight.
__inline__ __device__ float warpReduceSum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;   // valid on lane 0
}
__global__ void ivf_fine_kernel_warp(const float* __restrict__ Q, const float* __restrict__ RB,
                                     const int* __restrict__ coff, const int* __restrict__ idmap,
                                     const int* __restrict__ cids, int qn, int nprobe,
                                     int32_t* __restrict__ out_ids, float* __restrict__ out_val)
{
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int i    = gtid >> 5;            // one warp per query
    int lane = gtid & 31;
    if (i >= qn) return;

    const float* q = Q + (size_t)i * DIM;
    float q0 = q[lane], q1 = q[lane + 32], q2 = q[lane + 64];   // DIM = 96 = 3*32

    float bestVal[K]; int bestId[K];                            // maintained on lane 0
    #pragma unroll
    for (int t = 0; t < K; ++t) { bestVal[t] = -FLT_MAX; bestId[t] = -1; }

    for (int p = 0; p < nprobe; ++p) {
        int c = cids[(size_t)i * nprobe + p];
        int lo = coff[c], hi = coff[c + 1];
        for (int pos = lo; pos < hi; ++pos) {
            const float* b = RB + (size_t)pos * DIM;
            float partial = q0 * b[lane] + q1 * b[lane + 32] + q2 * b[lane + 64];
            float ip = warpReduceSum(partial);
            if (lane == 0 && ip > bestVal[K - 1]) {
                int pp = K - 1;
                while (pp > 0 && bestVal[pp - 1] < ip) { bestVal[pp] = bestVal[pp - 1]; bestId[pp] = bestId[pp - 1]; --pp; }
                bestVal[pp] = ip; bestId[pp] = idmap[pos];
            }
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (int t = 0; t < K; ++t) { out_ids[(size_t)i * K + t] = bestId[t]; out_val[(size_t)i * K + t] = bestVal[t]; }
    }
}


int main()
{
    print_device_info();
    printf("[cpu] OpenMP threads = %d\n", omp_get_max_threads());

    size_t n, bd, m, qd, gn, gt_d;
    float*   base  = load_data<float>  ("data/DEEP100K.base.100k.fbin",         n, bd);
    float*   query = load_data<float>  ("data/DEEP100K.query.fbin",             m, qd);
    int32_t* gt    = load_data<int32_t>("data/DEEP100K.gt.query.100k.top100.bin", gn, gt_d);
    printf("[data] base n=%zu d=%zu | query m=%zu\n", n, bd, m);

    float *d_base, *d_qchunk, *d_cent, *d_RB, *d_coarse, *d_val;
    int   *d_coff, *d_idmap, *d_cids;
    int32_t *d_ids;
    CUDA_CHECK(cudaMalloc(&d_base,   sizeof(float) * n * DIM));
    CUDA_CHECK(cudaMalloc(&d_qchunk, sizeof(float) * QCHUNK * DIM));
    CUDA_CHECK(cudaMalloc(&d_cent,   sizeof(float) * (size_t)MAX_NLIST * DIM));
    CUDA_CHECK(cudaMalloc(&d_RB,     sizeof(float) * n * DIM));
    CUDA_CHECK(cudaMalloc(&d_coff,   sizeof(int) * (MAX_NLIST + 1)));
    CUDA_CHECK(cudaMalloc(&d_idmap,  sizeof(int) * n));
    CUDA_CHECK(cudaMalloc(&d_coarse, sizeof(float) * (size_t)QCHUNK * MAX_NLIST));
    CUDA_CHECK(cudaMalloc(&d_cids,   sizeof(int) * (size_t)QCHUNK * MAXNPROBE));
    CUDA_CHECK(cudaMalloc(&d_ids,    sizeof(int32_t) * QCHUNK * K));
    CUDA_CHECK(cudaMalloc(&d_val,    sizeof(float) * QCHUNK * K));
    CUDA_CHECK(cudaMemcpy(d_base, base, sizeof(float) * n * DIM, cudaMemcpyHostToDevice));

    cublasHandle_t blas; cublasCreate(&blas);
    std::vector<int32_t> ids(m * K);
    const float alpha = 1.0f, beta = 0.0f;
    const int REP = 5;

    FILE* csv = fopen("results_ivf_opt.csv", "w");
    const char* header = "nlist,nprobe,recall,orig_total_us,opt_total_us,total_speedup,"
                         "orig_fine_us,opt_fine_us,fine_speedup";
    printf("\n%s\n", header); fprintf(csv, "%s\n", header);

    for (int li = 0; li < (int)(sizeof(NLIST_SWEEP) / sizeof(int)); ++li) {
        int nlist = NLIST_SWEEP[li];
        IVFIndex idx;
        char cache[128]; snprintf(cache, sizeof(cache), "data/ivf_nlist%d.cache", nlist);
        if (!load_ivf_cache(idx, cache, base, nlist)) {
            printf("[ivf] nlist=%d building ...\n", nlist);
            build_ivf(idx, base, (int)n, (int)DIM, nlist);
            save_ivf_cache(idx, cache);
        } else printf("[ivf] nlist=%d cache\n", nlist);
        CUDA_CHECK(cudaMemcpy(d_cent, idx.centroids.data(), sizeof(float) * (size_t)nlist * DIM, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_RB, idx.reordered_base.data(), sizeof(float) * n * DIM, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_coff, idx.cluster_off.data(), sizeof(int) * (nlist + 1), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_idmap, idx.id_map.data(), sizeof(int) * n, cudaMemcpyHostToDevice));

        // search; opt=false -> original kernel, opt=true -> warp kernel.
        // fine_ms accumulates ONLY the fine kernel time; total_ms wraps the pipeline.
        auto search = [&](int nprobe, bool opt, float& fine_ms) -> float {
            CudaTimer tt, tf; tt.start(); fine_ms = 0.0f;
            for (size_t q0 = 0; q0 < m; q0 += QCHUNK) {
                int qn = (int)std::min((size_t)QCHUNK, m - q0);
                CUDA_CHECK(cudaMemcpy(d_qchunk, query + q0 * DIM, sizeof(float) * qn * DIM, cudaMemcpyHostToDevice));
                cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, nlist, qn, DIM,
                            &alpha, d_cent, DIM, d_qchunk, DIM, &beta, d_coarse, nlist);
                select_topn_kernel<<<(qn + TPB - 1) / TPB, TPB>>>(d_coarse, qn, nlist, nprobe, d_cids);
                tf.start();
                if (!opt)
                    ivf_fine_kernel<<<(qn + TPB - 1) / TPB, TPB>>>(d_qchunk, d_RB, d_coff, d_idmap, d_cids, qn, nprobe, d_ids, d_val);
                else
                    ivf_fine_kernel_warp<<<(qn * 32 + TPB - 1) / TPB, TPB>>>(d_qchunk, d_RB, d_coff, d_idmap, d_cids, qn, nprobe, d_ids, d_val);
                fine_ms += tf.stop();
                CUDA_CHECK(cudaMemcpy(ids.data() + q0 * K, d_ids, sizeof(int32_t) * qn * K, cudaMemcpyDeviceToHost));
            }
            return tt.stop();
        };

        for (int pi = 0; pi < (int)(sizeof(NPROBE_SWEEP) / sizeof(int)); ++pi) {
            int np = NPROBE_SWEEP[pi];
            if (np > nlist) continue;
            float of, otf = 1e30f, ototal = 1e30f;          // original
            for (int r = 0; r < REP; ++r) { float t = search(np, false, of); if (t < ototal) { ototal = t; otf = of; } }
            double rec_o = compute_recall(ids.data(), gt, gt_d, m, K);

            float pf, ptf = 1e30f, ptotal = 1e30f;          // optimized
            for (int r = 0; r < REP; ++r) { float t = search(np, true, pf); if (t < ptotal) { ptotal = t; ptf = pf; } }
            double rec_p = compute_recall(ids.data(), gt, gt_d, m, K);

            char line[320];
            snprintf(line, sizeof(line), "%d,%d,%.5f/%.5f,%.3f,%.3f,%.2f,%.3f,%.3f,%.2f",
                     nlist, np, rec_o, rec_p,
                     1000.0 * ototal / m, 1000.0 * ptotal / m, ototal / ptotal,
                     1000.0 * otf / m, 1000.0 * ptf / m, otf / ptf);
            printf("%s\n", line); fprintf(csv, "%s\n", line); fflush(csv);
        }
    }
    fclose(csv);
    printf("\n[done] raw data written to results_ivf_opt.csv\n");

    cublasDestroy(blas);
    cudaFree(d_base); cudaFree(d_qchunk); cudaFree(d_cent); cudaFree(d_RB);
    cudaFree(d_coff); cudaFree(d_idmap); cudaFree(d_coarse); cudaFree(d_cids);
    cudaFree(d_ids); cudaFree(d_val);
    delete[] base; delete[] query; delete[] gt;
    return 0;
}
