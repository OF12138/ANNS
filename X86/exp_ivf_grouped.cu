// =============================================================================
// exp_ivf_grouped.cu -- IVF fine phase via the manual's GROUPING strategy
//                       (分组策略 + per-cluster GEMM), vs thread-scan & warp-scan
//
// The manual notes that with a query batch, different queries probe different
// clusters, so a single big matmul wastes work.  The fix it proposes: GROUP the
// batch's queries by the cluster they probe, then for each cluster do a dense
// matrix multiply  (cluster vectors) x (queries that probed it)  -> distances,
// reusing cuBLAS (near-peak GEMM) for the FINE phase too.
//
// Pipeline (whole m-query batch at once, no chunking):
//   1. coarse:  GEMM query x centroids -> select nprobe clusters per query.
//   2. GROUP :  invert (query -> clusters) into (cluster -> query list) with an
//               atomic histogram + prefix sum + scatter.            [分组策略]
//   3. fine  :  for each non-empty cluster c:
//                 gather its queries -> Qc ;  Sc = Bc · Qcᵀ (cuBLAS) ;
//                 merge Sc into each query's global top-k.
//
// Compared head-to-head with:
//   - thread-scan  ivf_fine_kernel        (baseline, 1 thread/query)
//   - warp-scan    ivf_fine_kernel_warp   (previous optimization, 1 warp/query)
// All three share coarse+select and produce the same recall.
//
//   >>> GROUPING CORE (report): group_count/scatter + per-cluster GEMM loop,
//       tagged "GROUP CORE".
//
// Build:  build.bat exp_ivf_grouped.cu exp_ivf_grouped.exe -Xcompiler /openmp -lcublas
// Run  :  ./exp_ivf_grouped
// =============================================================================

#include "gpu_utils.cuh"
#include <cfloat>
#include <cmath>
#include <random>
#include <cublas_v2.h>
#include <omp.h>

#define DIM        96
#define K          10
#define TPB        128
#define KM_ITERS   15
#define MAXNPROBE  64
#define MAX_NLIST  4096
#define SG_CAP     (32 * 1024 * 1024)   // d_Sg capacity in floats (qc*sz must fit)

static const int NLIST_SWEEP[]  = {256, 1024, 4096};
static const int NPROBE_SWEEP[] = {1, 2, 4, 8, 16, 32, 64};


// ── IVF index + build + cache (same format as exp_ivf_pareto.cu) ─────────────
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


// ── coarse cluster selection ─────────────────────────────────────────────────
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

// ── thread-scan (baseline) and warp-scan (prev opt) fine kernels ─────────────
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
__inline__ __device__ float warpReduceSum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}
__global__ void ivf_fine_kernel_warp(const float* __restrict__ Q, const float* __restrict__ RB,
                                     const int* __restrict__ coff, const int* __restrict__ idmap,
                                     const int* __restrict__ cids, int qn, int nprobe,
                                     int32_t* __restrict__ out_ids, float* __restrict__ out_val)
{
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = gtid >> 5, lane = gtid & 31;
    if (i >= qn) return;
    const float* q = Q + (size_t)i * DIM;
    float q0 = q[lane], q1 = q[lane + 32], q2 = q[lane + 64];
    float bestVal[K]; int bestId[K];
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
    if (lane == 0)
        #pragma unroll
        for (int t = 0; t < K; ++t) { out_ids[(size_t)i * K + t] = bestId[t]; out_val[(size_t)i * K + t] = bestVal[t]; }
}


// ── GROUP CORE kernels: build (cluster -> query list) and merge GEMM results ──
__global__ void group_count_kernel(const int* __restrict__ cids, int total, int* __restrict__ qcount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    atomicAdd(&qcount[cids[idx]], 1);
}
__global__ void group_scatter_kernel(const int* __restrict__ cids, int total, int nprobe,
                                     int* __restrict__ cursor, int* __restrict__ group_q)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int qid = idx / nprobe;                  // which query
    int c   = cids[idx];                     // which cluster
    int pos = atomicAdd(&cursor[c], 1);      // its slot in cluster c's list
    group_q[pos] = qid;
}
__global__ void init_topk_kernel(float* topval, int* topid, int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    #pragma unroll
    for (int t = 0; t < K; ++t) { topval[(size_t)i * K + t] = -FLT_MAX; topid[(size_t)i * K + t] = -1; }
}
__global__ void gather_queries_kernel(const float* __restrict__ Q, const int* __restrict__ group_q,
                                      int goff, int qc, float* __restrict__ Qbuf)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= qc) return;
    int qid = group_q[goff + r];
    #pragma unroll
    for (int t = 0; t < DIM; ++t) Qbuf[(size_t)r * DIM + t] = Q[(size_t)qid * DIM + t];
}
// merge one cluster's GEMM block Sc (qc x sz, row-major) into each query's top-k.
// Each query appears once per cluster and clusters run sequentially -> no races.
__global__ void group_update_kernel(const float* __restrict__ Sg, const int* __restrict__ group_q,
                                    int goff, int qc, int sz, int base_off,
                                    const int* __restrict__ idmap,
                                    float* __restrict__ topval, int* __restrict__ topid)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= qc) return;
    int qid = group_q[goff + r];
    float* bv = topval + (size_t)qid * K;
    int*   bi = topid  + (size_t)qid * K;
    const float* row = Sg + (size_t)r * sz;
    for (int j = 0; j < sz; ++j) {
        float v = row[j];
        if (v > bv[K - 1]) {
            int p = K - 1;
            while (p > 0 && bv[p - 1] < v) { bv[p] = bv[p - 1]; bi[p] = bi[p - 1]; --p; }
            bv[p] = v; bi[p] = idmap[base_off + j];
        }
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

    float *d_query, *d_cent, *d_RB, *d_coarse, *d_Qbuf, *d_Sg, *d_topval, *d_val;
    int   *d_coff, *d_idmap, *d_cids, *d_qcount, *d_cursor, *d_group_q;
    int32_t *d_topid;
    CUDA_CHECK(cudaMalloc(&d_query,  sizeof(float) * m * DIM));
    CUDA_CHECK(cudaMalloc(&d_cent,   sizeof(float) * (size_t)MAX_NLIST * DIM));
    CUDA_CHECK(cudaMalloc(&d_RB,     sizeof(float) * n * DIM));
    CUDA_CHECK(cudaMalloc(&d_coarse, sizeof(float) * (size_t)MAX_NLIST * m));
    CUDA_CHECK(cudaMalloc(&d_Qbuf,   sizeof(float) * m * DIM));
    CUDA_CHECK(cudaMalloc(&d_Sg,     sizeof(float) * (size_t)SG_CAP));
    CUDA_CHECK(cudaMalloc(&d_topval, sizeof(float) * m * K));
    CUDA_CHECK(cudaMalloc(&d_val,    sizeof(float) * m * K));
    CUDA_CHECK(cudaMalloc(&d_coff,   sizeof(int) * (MAX_NLIST + 1)));
    CUDA_CHECK(cudaMalloc(&d_idmap,  sizeof(int) * n));
    CUDA_CHECK(cudaMalloc(&d_cids,   sizeof(int) * m * MAXNPROBE));
    CUDA_CHECK(cudaMalloc(&d_qcount, sizeof(int) * MAX_NLIST));
    CUDA_CHECK(cudaMalloc(&d_cursor, sizeof(int) * MAX_NLIST));
    CUDA_CHECK(cudaMalloc(&d_group_q, sizeof(int) * m * MAXNPROBE));
    CUDA_CHECK(cudaMalloc(&d_topid,  sizeof(int32_t) * m * K));
    CUDA_CHECK(cudaMemcpy(d_query, query, sizeof(float) * m * DIM, cudaMemcpyHostToDevice));

    cublasHandle_t blas; cublasCreate(&blas);
    std::vector<int32_t> ids(m * K);
    std::vector<int>     h_qcount(MAX_NLIST), h_goff(MAX_NLIST + 1);
    const float alpha = 1.0f, beta = 0.0f;
    const int REP = 3;

    FILE* csv = fopen("results_ivf_grouped.csv", "w");
    const char* header = "nlist,nprobe,recall,thread_us,warp_us,grouped_us,"
                         "grouped_vs_thread,grouped_vs_warp";
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
        const std::vector<int>& h_coff = idx.cluster_off;
        CUDA_CHECK(cudaMemcpy(d_cent, idx.centroids.data(), sizeof(float) * (size_t)nlist * DIM, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_RB, idx.reordered_base.data(), sizeof(float) * n * DIM, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_coff, h_coff.data(), sizeof(int) * (nlist + 1), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_idmap, idx.id_map.data(), sizeof(int) * n, cudaMemcpyHostToDevice));

        // coarse + select shared by all methods; writes d_cids for the whole batch
        auto coarse = [&](int nprobe) {
            cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, nlist, (int)m, DIM,
                        &alpha, d_cent, DIM, d_query, DIM, &beta, d_coarse, nlist);
            select_topn_kernel<<<((int)m + TPB - 1) / TPB, TPB>>>(d_coarse, (int)m, nlist, nprobe, d_cids);
        };
        // thread-scan / warp-scan full-batch fine
        auto run_scan = [&](int nprobe, bool warp) -> float {
            CudaTimer tt; tt.start();
            coarse(nprobe);
            if (!warp) ivf_fine_kernel<<<((int)m + TPB - 1) / TPB, TPB>>>(d_query, d_RB, d_coff, d_idmap, d_cids, (int)m, nprobe, d_topid, d_val);
            else       ivf_fine_kernel_warp<<<((int)m * 32 + TPB - 1) / TPB, TPB>>>(d_query, d_RB, d_coff, d_idmap, d_cids, (int)m, nprobe, d_topid, d_val);
            CUDA_CHECK(cudaMemcpy(ids.data(), d_topid, sizeof(int32_t) * m * K, cudaMemcpyDeviceToHost));
            return tt.stop();
        };
        // ── GROUP CORE: grouping + per-cluster GEMM fine ─────────────────────
        auto run_grouped = [&](int nprobe) -> float {
            int total = (int)m * nprobe;
            CudaTimer tt; tt.start();
            coarse(nprobe);
            // (a) histogram queries per cluster
            CUDA_CHECK(cudaMemset(d_qcount, 0, sizeof(int) * nlist));
            group_count_kernel<<<(total + TPB - 1) / TPB, TPB>>>(d_cids, total, d_qcount);
            // (b) prefix sum on host -> group offsets, reset cursor=offsets
            CUDA_CHECK(cudaMemcpy(h_qcount.data(), d_qcount, sizeof(int) * nlist, cudaMemcpyDeviceToHost));
            h_goff[0] = 0;
            for (int c = 0; c < nlist; ++c) h_goff[c + 1] = h_goff[c] + h_qcount[c];
            CUDA_CHECK(cudaMemcpy(d_cursor, h_goff.data(), sizeof(int) * nlist, cudaMemcpyHostToDevice));
            // (c) scatter query ids into (cluster -> query list)
            group_scatter_kernel<<<(total + TPB - 1) / TPB, TPB>>>(d_cids, total, nprobe, d_cursor, d_group_q);
            // (d) init global top-k
            init_topk_kernel<<<((int)m + TPB - 1) / TPB, TPB>>>(d_topval, d_topid, (int)m);
            // (e) per-cluster: gather queries -> GEMM Bc·Qcᵀ -> merge top-k
            for (int c = 0; c < nlist; ++c) {
                int qc = h_goff[c + 1] - h_goff[c];
                int sz = h_coff[c + 1] - h_coff[c];
                if (qc == 0 || sz == 0) continue;
                if ((size_t)qc * sz > SG_CAP) continue;     // safety (never hit in sweep)
                int goff = h_goff[c];
                gather_queries_kernel<<<(qc + TPB - 1) / TPB, TPB>>>(d_query, d_group_q, goff, qc, d_Qbuf);
                cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, sz, qc, DIM,
                            &alpha, d_RB + (size_t)h_coff[c] * DIM, DIM, d_Qbuf, DIM, &beta, d_Sg, sz);
                group_update_kernel<<<(qc + TPB - 1) / TPB, TPB>>>(d_Sg, d_group_q, goff, qc, sz, h_coff[c], d_idmap, d_topval, d_topid);
            }
            CUDA_CHECK(cudaMemcpy(ids.data(), d_topid, sizeof(int32_t) * m * K, cudaMemcpyDeviceToHost));
            return tt.stop();
        };

        for (int pi = 0; pi < (int)(sizeof(NPROBE_SWEEP) / sizeof(int)); ++pi) {
            int np = NPROBE_SWEEP[pi];
            if (np > nlist) continue;
            float t_thread = 1e30f; for (int r = 0; r < REP; ++r) t_thread = std::min(t_thread, run_scan(np, false));
            float t_warp   = 1e30f; for (int r = 0; r < REP; ++r) t_warp   = std::min(t_warp,   run_scan(np, true));
            float t_group  = 1e30f; for (int r = 0; r < REP; ++r) t_group  = std::min(t_group,  run_grouped(np));
            double rec = compute_recall(ids.data(), gt, gt_d, m, K);   // recall of grouped (== others)

            double us = 1000.0 / m;   // ms-total -> us/query factor
            char line[320];
            snprintf(line, sizeof(line), "%d,%d,%.5f,%.3f,%.3f,%.3f,%.2f,%.2f",
                     nlist, np, rec, t_thread * us, t_warp * us, t_group * us,
                     t_thread / t_group, t_warp / t_group);
            printf("%s\n", line); fprintf(csv, "%s\n", line); fflush(csv);
        }
    }
    fclose(csv);
    printf("\n[done] raw data written to results_ivf_grouped.csv\n");

    cublasDestroy(blas);
    cudaFree(d_query); cudaFree(d_cent); cudaFree(d_RB); cudaFree(d_coarse);
    cudaFree(d_Qbuf); cudaFree(d_Sg); cudaFree(d_topval); cudaFree(d_val);
    cudaFree(d_coff); cudaFree(d_idmap); cudaFree(d_cids); cudaFree(d_qcount);
    cudaFree(d_cursor); cudaFree(d_group_q); cudaFree(d_topid);
    delete[] base; delete[] query; delete[] gt;
    return 0;
}
