// =============================================================================
// exp_ivf_grouped_batched.cu -- BATCHED grouping: fuse the per-cluster GEMM
//                               loop into ONE kernel launch
//
// exp_ivf_grouped.cu showed the grouping strategy (cluster -> query inversion +
// one dense cuBLAS GEMM per cluster) wins at small nlist / large nprobe but is
// crushed by kernel-launch overhead at large nlist: the fine phase issues
// 3*nlist tiny launches (gather + GEMM + merge per cluster), each costing more
// than the math it performs.  CUDA 12.2's cuBLAS has no grouped/variable-size
// batched SGEMM, so this experiment implements the batching by hand:
//
//   fine phase = ONE launch of `batched_fine_kernel`
//     - host chops every cluster's (query-group x vectors) block into 32-query
//       tiles; one thread block processes one tile: it stages the 32 queries
//       and 32-vector strips of the cluster in shared memory (a hand-written
//       tiled GEMM), and keeps each query-slot's running top-k in registers.
//     - a second kernel merges each query's nprobe slot top-k lists.
//   So the 3*nlist launches collapse to 2, independent of nlist.
//
// Compared head-to-head (same coarse + select, identical recall by design):
//   thread-scan / warp-scan / grouped (per-cluster cuBLAS) / batched (this)
//
// Build:  build.bat exp_ivf_grouped_batched.cu exp_ivf_grouped_batched.exe -Xcompiler /openmp -lcublas
// Run  :  ./exp_ivf_grouped_batched
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
#define SG_CAP     (32 * 1024 * 1024)   // d_Sg capacity in floats (grouped path)
#define TQ         32                   // queries per tile (batched path)
#define TB         32                   // base vectors per shared-memory strip

static const int NLIST_SWEEP[]  = {256, 1024, 4096};
static const int NPROBE_SWEEP[] = {1, 2, 4, 8, 16, 32, 64};


// ── IVF index + build + cache (same format as exp_ivf_grouped.cu) ────────────
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

// ── thread-scan / warp-scan fine kernels (baselines, unchanged) ──────────────
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


// ── GROUP kernels (identical logic to exp_ivf_grouped.cu) ────────────────────
__global__ void group_count_kernel(const int* __restrict__ cids, int total, int* __restrict__ qcount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    atomicAdd(&qcount[cids[idx]], 1);
}
// scatter, plus the inverse map slot_map[(query,probe)] -> group position that
// the batched merge kernel needs.  (grouped path simply ignores slot_map.)
__global__ void group_scatter_map_kernel(const int* __restrict__ cids, int total, int nprobe,
                                         int* __restrict__ cursor, int* __restrict__ group_q,
                                         int* __restrict__ slot_map)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int qid = idx / nprobe;
    int c   = cids[idx];
    int pos = atomicAdd(&cursor[c], 1);
    group_q[pos]  = qid;
    slot_map[idx] = pos;
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


// ── BATCH CORE 1: one block = one (cluster, 32-query tile); the whole fine ───
// phase is a single launch.  tiles[b] = {cluster c, absolute group row grow0,
// row count qcnt, pad}.  Shared-memory tiled GEMM (sQ 12KB + sB 12KB + sS 4KB);
// each of the first qcnt threads owns one query row and keeps its slot's
// running top-k in registers across the cluster's TB-vector strips.
__global__ void batched_fine_kernel(const float* __restrict__ Q, const float* __restrict__ RB,
                                    const int* __restrict__ coff, const int* __restrict__ idmap,
                                    const int* __restrict__ group_q, const int4* __restrict__ tiles,
                                    float* __restrict__ slot_val, int32_t* __restrict__ slot_id)
{
    __shared__ float sQ[TQ][DIM];
    __shared__ float sB[TB][DIM];
    __shared__ float sS[TQ][TB];
    __shared__ int   sQid[TQ];
    int4 tile = tiles[blockIdx.x];
    int c = tile.x, grow0 = tile.y, qcnt = tile.z;
    int tid = threadIdx.x;                            // blockDim.x == 256
    if (tid < qcnt) sQid[tid] = group_q[grow0 + tid];
    __syncthreads();
    for (int idx = tid; idx < qcnt * DIM; idx += 256) {
        int r = idx / DIM, t = idx % DIM;
        sQ[r][t] = Q[(size_t)sQid[r] * DIM + t];
    }
    int lo = coff[c], hi = coff[c + 1];
    float bestVal[K]; int bestId[K];                  // row tid's running top-k
    #pragma unroll
    for (int t = 0; t < K; ++t) { bestVal[t] = -FLT_MAX; bestId[t] = -1; }
    for (int pos0 = lo; pos0 < hi; pos0 += TB) {
        int bcnt = min(TB, hi - pos0);
        __syncthreads();                              // sB reuse guard
        for (int idx = tid; idx < bcnt * DIM; idx += 256) {
            int j = idx / DIM, t = idx % DIM;
            sB[j][t] = RB[(size_t)(pos0 + j) * DIM + t];
        }
        __syncthreads();
        for (int pair = tid; pair < TQ * TB; pair += 256) {   // 4 dots per thread
            int r = pair / TB, j = pair % TB;
            float acc = -FLT_MAX;
            if (r < qcnt && j < bcnt) {
                acc = 0.0f;
                #pragma unroll
                for (int t = 0; t < DIM; ++t) acc += sQ[r][t] * sB[j][t];
            }
            sS[r][j] = acc;
        }
        __syncthreads();
        if (tid < qcnt) {                             // row owner merges strip
            for (int j = 0; j < bcnt; ++j) {
                float v = sS[tid][j];
                if (v > bestVal[K - 1]) {
                    int pp = K - 1;
                    while (pp > 0 && bestVal[pp - 1] < v) { bestVal[pp] = bestVal[pp - 1]; bestId[pp] = bestId[pp - 1]; --pp; }
                    bestVal[pp] = v; bestId[pp] = idmap[pos0 + j];
                }
            }
        }
    }
    if (tid < qcnt) {
        size_t off = (size_t)(grow0 + tid) * K;
        #pragma unroll
        for (int t = 0; t < K; ++t) { slot_val[off + t] = bestVal[t]; slot_id[off + t] = bestId[t]; }
    }
}

// ── BATCH CORE 2: per-query merge of its nprobe slot top-k lists ─────────────
__global__ void batched_merge_kernel(const float* __restrict__ slot_val, const int32_t* __restrict__ slot_id,
                                     const int* __restrict__ slot_map, int qn, int nprobe,
                                     int32_t* __restrict__ out_ids)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= qn) return;
    float bv[K]; int bi[K];
    #pragma unroll
    for (int t = 0; t < K; ++t) { bv[t] = -FLT_MAX; bi[t] = -1; }
    for (int p = 0; p < nprobe; ++p) {
        size_t off = (size_t)slot_map[(size_t)i * nprobe + p] * K;
        for (int t = 0; t < K; ++t) {
            float v = slot_val[off + t];
            if (v > bv[K - 1]) {
                int pp = K - 1;
                while (pp > 0 && bv[pp - 1] < v) { bv[pp] = bv[pp - 1]; bi[pp] = bi[pp - 1]; --pp; }
                bv[pp] = v; bi[pp] = slot_id[off + t];
            }
        }
    }
    #pragma unroll
    for (int t = 0; t < K; ++t) out_ids[(size_t)i * K + t] = bi[t];
}


int main()
{
    print_device_info();
    printf("[cpu] OpenMP threads = %d\n", omp_get_max_threads());

    size_t n, bd, m, qd, gn, gt_d;
    float*   base  = load_data<float>  ("data/DEEP100K.base.100k.fbin",           n, bd);
    float*   query = load_data<float>  ("data/DEEP100K.query.fbin",               m, qd);
    int32_t* gt    = load_data<int32_t>("data/DEEP100K.gt.query.100k.top100.bin", gn, gt_d);
    printf("[data] base n=%zu d=%zu | query m=%zu\n", n, bd, m);

    const int MAX_TILES = (int)(m * MAXNPROBE / TQ) + MAX_NLIST + 16;
    float *d_query, *d_cent, *d_RB, *d_coarse, *d_Qbuf, *d_Sg, *d_topval, *d_val, *d_slotval;
    int   *d_coff, *d_idmap, *d_cids, *d_qcount, *d_cursor, *d_group_q, *d_slotmap;
    int32_t *d_topid, *d_slotid;
    int4  *d_tiles;
    CUDA_CHECK(cudaMalloc(&d_query,  sizeof(float) * m * DIM));
    CUDA_CHECK(cudaMalloc(&d_cent,   sizeof(float) * (size_t)MAX_NLIST * DIM));
    CUDA_CHECK(cudaMalloc(&d_RB,     sizeof(float) * n * DIM));
    CUDA_CHECK(cudaMalloc(&d_coarse, sizeof(float) * (size_t)MAX_NLIST * m));
    CUDA_CHECK(cudaMalloc(&d_Qbuf,   sizeof(float) * m * DIM));
    CUDA_CHECK(cudaMalloc(&d_Sg,     sizeof(float) * (size_t)SG_CAP));
    CUDA_CHECK(cudaMalloc(&d_topval, sizeof(float) * m * K));
    CUDA_CHECK(cudaMalloc(&d_val,    sizeof(float) * m * K));
    CUDA_CHECK(cudaMalloc(&d_slotval, sizeof(float) * m * MAXNPROBE * K));
    CUDA_CHECK(cudaMalloc(&d_slotid,  sizeof(int32_t) * m * MAXNPROBE * K));
    CUDA_CHECK(cudaMalloc(&d_coff,   sizeof(int) * (MAX_NLIST + 1)));
    CUDA_CHECK(cudaMalloc(&d_idmap,  sizeof(int) * n));
    CUDA_CHECK(cudaMalloc(&d_cids,   sizeof(int) * m * MAXNPROBE));
    CUDA_CHECK(cudaMalloc(&d_qcount, sizeof(int) * MAX_NLIST));
    CUDA_CHECK(cudaMalloc(&d_cursor, sizeof(int) * MAX_NLIST));
    CUDA_CHECK(cudaMalloc(&d_group_q, sizeof(int) * m * MAXNPROBE));
    CUDA_CHECK(cudaMalloc(&d_slotmap, sizeof(int) * m * MAXNPROBE));
    CUDA_CHECK(cudaMalloc(&d_tiles,  sizeof(int4) * MAX_TILES));
    CUDA_CHECK(cudaMalloc(&d_topid,  sizeof(int32_t) * m * K));
    CUDA_CHECK(cudaMemcpy(d_query, query, sizeof(float) * m * DIM, cudaMemcpyHostToDevice));

    cublasHandle_t blas; cublasCreate(&blas);
    std::vector<int32_t> ids(m * K);
    std::vector<int>     h_qcount(MAX_NLIST), h_goff(MAX_NLIST + 1);
    std::vector<int4>    h_tiles(MAX_TILES);
    const float alpha = 1.0f, beta = 0.0f;
    const int REP = 3;

    FILE* csv = fopen("results_ivf_grouped_batched.csv", "w");
    const char* header = "nlist,nprobe,recall,thread_us,warp_us,grouped_us,batched_us,"
                         "batched_vs_grouped,batched_vs_warp,ntiles";
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

        auto coarse = [&](int nprobe) {
            cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, nlist, (int)m, DIM,
                        &alpha, d_cent, DIM, d_query, DIM, &beta, d_coarse, nlist);
            select_topn_kernel<<<((int)m + TPB - 1) / TPB, TPB>>>(d_coarse, (int)m, nlist, nprobe, d_cids);
        };
        // shared grouping stage (count -> host prefix sum -> scatter+map)
        auto do_group = [&](int nprobe) {
            int total = (int)m * nprobe;
            CUDA_CHECK(cudaMemset(d_qcount, 0, sizeof(int) * nlist));
            group_count_kernel<<<(total + TPB - 1) / TPB, TPB>>>(d_cids, total, d_qcount);
            CUDA_CHECK(cudaMemcpy(h_qcount.data(), d_qcount, sizeof(int) * nlist, cudaMemcpyDeviceToHost));
            h_goff[0] = 0;
            for (int c = 0; c < nlist; ++c) h_goff[c + 1] = h_goff[c] + h_qcount[c];
            CUDA_CHECK(cudaMemcpy(d_cursor, h_goff.data(), sizeof(int) * nlist, cudaMemcpyHostToDevice));
            group_scatter_map_kernel<<<(total + TPB - 1) / TPB, TPB>>>(d_cids, total, nprobe, d_cursor, d_group_q, d_slotmap);
        };
        auto run_scan = [&](int nprobe, bool warp) -> float {
            CudaTimer tt; tt.start();
            coarse(nprobe);
            if (!warp) ivf_fine_kernel<<<((int)m + TPB - 1) / TPB, TPB>>>(d_query, d_RB, d_coff, d_idmap, d_cids, (int)m, nprobe, d_topid, d_val);
            else       ivf_fine_kernel_warp<<<((int)m * 32 + TPB - 1) / TPB, TPB>>>(d_query, d_RB, d_coff, d_idmap, d_cids, (int)m, nprobe, d_topid, d_val);
            CUDA_CHECK(cudaMemcpy(ids.data(), d_topid, sizeof(int32_t) * m * K, cudaMemcpyDeviceToHost));
            return tt.stop();
        };
        // grouped: per-cluster cuBLAS loop (identical to exp_ivf_grouped.cu)
        auto run_grouped = [&](int nprobe) -> float {
            CudaTimer tt; tt.start();
            coarse(nprobe);
            do_group(nprobe);
            init_topk_kernel<<<((int)m + TPB - 1) / TPB, TPB>>>(d_topval, d_topid, (int)m);
            for (int c = 0; c < nlist; ++c) {
                int qc = h_goff[c + 1] - h_goff[c];
                int sz = h_coff[c + 1] - h_coff[c];
                if (qc == 0 || sz == 0) continue;
                if ((size_t)qc * sz > SG_CAP) continue;
                int goff = h_goff[c];
                gather_queries_kernel<<<(qc + TPB - 1) / TPB, TPB>>>(d_query, d_group_q, goff, qc, d_Qbuf);
                cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, sz, qc, DIM,
                            &alpha, d_RB + (size_t)h_coff[c] * DIM, DIM, d_Qbuf, DIM, &beta, d_Sg, sz);
                group_update_kernel<<<(qc + TPB - 1) / TPB, TPB>>>(d_Sg, d_group_q, goff, qc, sz, h_coff[c], d_idmap, d_topval, d_topid);
            }
            CUDA_CHECK(cudaMemcpy(ids.data(), d_topid, sizeof(int32_t) * m * K, cudaMemcpyDeviceToHost));
            return tt.stop();
        };
        // ── BATCHED: tile list -> ONE fine launch + ONE merge launch ─────────
        int last_ntiles = 0;
        auto run_batched = [&](int nprobe) -> float {
            CudaTimer tt; tt.start();
            coarse(nprobe);
            do_group(nprobe);
            int ntiles = 0;                       // host-side tile chop
            for (int c = 0; c < nlist; ++c) {
                int qc = h_goff[c + 1] - h_goff[c];
                int sz = h_coff[c + 1] - h_coff[c];
                if (qc == 0 || sz == 0) continue;
                for (int r0 = 0; r0 < qc; r0 += TQ) {
                    h_tiles[ntiles].x = c;
                    h_tiles[ntiles].y = h_goff[c] + r0;
                    h_tiles[ntiles].z = std::min(TQ, qc - r0);
                    h_tiles[ntiles].w = 0;
                    ++ntiles;
                }
            }
            last_ntiles = ntiles;
            CUDA_CHECK(cudaMemcpy(d_tiles, h_tiles.data(), sizeof(int4) * ntiles, cudaMemcpyHostToDevice));
            batched_fine_kernel<<<ntiles, 256>>>(d_query, d_RB, d_coff, d_idmap,
                                                 d_group_q, d_tiles, d_slotval, d_slotid);
            batched_merge_kernel<<<((int)m + TPB - 1) / TPB, TPB>>>(d_slotval, d_slotid, d_slotmap,
                                                                    (int)m, nprobe, d_topid);
            CUDA_CHECK(cudaMemcpy(ids.data(), d_topid, sizeof(int32_t) * m * K, cudaMemcpyDeviceToHost));
            return tt.stop();
        };

        for (int pi = 0; pi < (int)(sizeof(NPROBE_SWEEP) / sizeof(int)); ++pi) {
            int np = NPROBE_SWEEP[pi];
            if (np > nlist) continue;
            float t_thread = 1e30f; for (int r = 0; r < REP; ++r) t_thread = std::min(t_thread, run_scan(np, false));
            float t_warp   = 1e30f; for (int r = 0; r < REP; ++r) t_warp   = std::min(t_warp,   run_scan(np, true));
            float t_group  = 1e30f; for (int r = 0; r < REP; ++r) t_group  = std::min(t_group,  run_grouped(np));
            double rec_g = compute_recall(ids.data(), gt, gt_d, m, K);
            float t_batch  = 1e30f; for (int r = 0; r < REP; ++r) t_batch  = std::min(t_batch,  run_batched(np));
            double rec_b = compute_recall(ids.data(), gt, gt_d, m, K);
            if (std::abs(rec_g - rec_b) > 1e-9)
                printf("[warn] recall mismatch grouped=%.6f batched=%.6f\n", rec_g, rec_b);

            double us = 1000.0 / m;
            char line[360];
            snprintf(line, sizeof(line), "%d,%d,%.5f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%d",
                     nlist, np, rec_b, t_thread * us, t_warp * us, t_group * us, t_batch * us,
                     t_group / t_batch, t_warp / t_batch, last_ntiles);
            printf("%s\n", line); fprintf(csv, "%s\n", line); fflush(csv);
        }
    }
    fclose(csv);
    printf("\n[done] raw data written to results_ivf_grouped_batched.csv\n");

    cublasDestroy(blas);
    cudaFree(d_query); cudaFree(d_cent); cudaFree(d_RB); cudaFree(d_coarse);
    cudaFree(d_Qbuf); cudaFree(d_Sg); cudaFree(d_topval); cudaFree(d_val);
    cudaFree(d_slotval); cudaFree(d_slotid); cudaFree(d_coff); cudaFree(d_idmap);
    cudaFree(d_cids); cudaFree(d_qcount); cudaFree(d_cursor); cudaFree(d_group_q);
    cudaFree(d_slotmap); cudaFree(d_tiles); cudaFree(d_topid);
    delete[] base; delete[] query; delete[] gt;
    return 0;
}
