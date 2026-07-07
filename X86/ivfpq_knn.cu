// =============================================================================
// ivfpq_knn.cu -- GPU IVF-PQ: IVF coarse pruning + residual PQ (ADC) fine scan
//                 + exact float rerank.  New file; ivf_knn.cu is untouched.
//
// Index (offline, cached to data/ivfpq_nlist<N>_M8.cache):
//   1. IVF k-means (same build/cache as exp_ivf_grouped.cu, cluster-reordered).
//   2. residual r_i = x_i - centroid(x_i); per-subspace k-means (M=8, K=256,
//      dsub=12) on residuals; codes stored in reordered-position order, so the
//      fine phase reads 8 B/vector instead of 384 B (48x memory compression).
//
// Query (online, batch of QCHUNK queries at a time):
//   coarse : cuBLAS GEMM query x centroids + select top-nprobe   [reused]
//   LUT    : one kernel computes LUT[slot][m][j] = <q - cent_c, pqcent[m][j]>
//            for every (query, probe) slot; bias <q, cent_c> comes free from
//            the coarse score matrix.
//   scan   : one WARP per (query, probe) slot; LUT staged in shared memory;
//            each lane scans a strided slice of the cluster's codes, lane 0
//            keeps the slot's top-p (ADC approximate scores).
//   merge  : one thread per query merges its nprobe slot lists -> global top-p
//            (union of per-cluster top-p is a superset of the global top-p,
//            so this equals the CPU version's global heap semantics).
//   rerank : one warp per query recomputes exact IP for the p candidates on
//            the original float vectors -> final top-k.
//
// Output: results_ivfpq.csv  (per config: recall, us/query, phase breakdown,
//         plus an exact warp-scan IVF run at the same nlist/nprobe as baseline)
//
// Build:  build.bat ivfpq_knn.cu ivfpq_knn.exe -Xcompiler /openmp -lcublas
// Run  :  ./ivfpq_knn        (from X86/, expects data/ + existing ivf caches)
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
#define PQ_M       8
#define PQ_K       256
#define PQ_DSUB    (DIM / PQ_M)          // 12
#define PQ_ITERS   15
#define MAXNPROBE  64
#define MAX_NLIST  4096
#define MAX_P      512                   // rerank depth upper bound (multiple of 32)
#define MAX_LANE_L (MAX_P / 32)          // per-lane kept candidates upper bound
#define QCHUNK     1024                  // queries per GPU pass (bounds buffers)

static const int NLIST_SWEEP[]  = {256, 1024};
static const int NPROBE_SWEEP[] = {2, 4, 8, 16, 32, 64};
static const int P_SWEEP[]      = {32, 128, 512};   // rerank set size (32-aligned)

// ── IVF index + build + cache (same format/logic as exp_ivf_grouped.cu) ──────
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

// ── residual PQ: train on residuals, encode reordered base ───────────────────
// pqcent: [M][K][dsub]  codes: [n][M] uint8, row `pos` matches reordered_base
struct PQCodebook {
    std::vector<float>   pqcent;   // M*K*dsub
    std::vector<uint8_t> codes;    // n*M
};
static void build_residual_pq(PQCodebook& pq, const IVFIndex& idx)
{
    const int n = idx.n, nlist = idx.nlist;
    // residuals of the reordered base (pos order)
    std::vector<float> resid((size_t)n * DIM);
    #pragma omp parallel for schedule(static)
    for (int pos = 0; pos < n; ++pos) {
        // find cluster of `pos` by offset table (clusters are contiguous)
        // linear scan is fine offline, but use upper_bound for speed:
        int lo = 0, hi = nlist;
        while (lo + 1 < hi) { int mid = (lo + hi) / 2; if (idx.cluster_off[mid] <= pos) lo = mid; else hi = mid; }
        const float* ce = idx.centroids.data() + (size_t)lo * DIM;
        const float* x  = idx.reordered_base.data() + (size_t)pos * DIM;
        float* r = resid.data() + (size_t)pos * DIM;
        for (int t = 0; t < DIM; ++t) r[t] = x[t] - ce[t];
    }
    pq.pqcent.assign((size_t)PQ_M * PQ_K * PQ_DSUB, 0.0f);
    pq.codes.assign((size_t)n * PQ_M, 0);
    // per-subspace k-means on residuals (L2)
    #pragma omp parallel for schedule(dynamic)
    for (int m = 0; m < PQ_M; ++m) {
        float* cent = pq.pqcent.data() + (size_t)m * PQ_K * PQ_DSUB;
        std::mt19937 rng(1234 + m);
        std::vector<int> perm(n);
        for (int i = 0; i < n; ++i) perm[i] = i;
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int j = 0; j < PQ_K; ++j)
            for (int t = 0; t < PQ_DSUB; ++t)
                cent[(size_t)j * PQ_DSUB + t] = resid[(size_t)perm[j] * DIM + m * PQ_DSUB + t];
        std::vector<int> code(n, 0);
        for (int it = 0; it < PQ_ITERS; ++it) {
            for (int i = 0; i < n; ++i) {                       // assign (L2)
                const float* r = resid.data() + (size_t)i * DIM + m * PQ_DSUB;
                float best = FLT_MAX; int bj = 0;
                for (int j = 0; j < PQ_K; ++j) {
                    const float* ce = cent + (size_t)j * PQ_DSUB;
                    float d2 = 0.0f;
                    for (int t = 0; t < PQ_DSUB; ++t) { float df = r[t] - ce[t]; d2 += df * df; }
                    if (d2 < best) { best = d2; bj = j; }
                }
                code[i] = bj;
            }
            std::vector<double> sum((size_t)PQ_K * PQ_DSUB, 0.0);
            std::vector<int>    cnt(PQ_K, 0);
            for (int i = 0; i < n; ++i) {
                const float* r = resid.data() + (size_t)i * DIM + m * PQ_DSUB;
                double* s = sum.data() + (size_t)code[i] * PQ_DSUB;
                for (int t = 0; t < PQ_DSUB; ++t) s[t] += r[t];
                cnt[code[i]]++;
            }
            for (int j = 0; j < PQ_K; ++j) {
                if (cnt[j] == 0) continue;                       // keep old centroid
                for (int t = 0; t < PQ_DSUB; ++t)
                    cent[(size_t)j * PQ_DSUB + t] = (float)(sum[(size_t)j * PQ_DSUB + t] / cnt[j]);
            }
        }
        for (int i = 0; i < n; ++i) pq.codes[(size_t)i * PQ_M + m] = (uint8_t)code[i];
    }
}
static bool load_pq_cache(PQCodebook& pq, const char* path, int n)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;
    int hm, hk, hd, hn;
    f.read((char*)&hm, 4); f.read((char*)&hk, 4); f.read((char*)&hd, 4); f.read((char*)&hn, 4);
    if (hm != PQ_M || hk != PQ_K || hd != PQ_DSUB || hn != n) return false;
    pq.pqcent.resize((size_t)PQ_M * PQ_K * PQ_DSUB);
    pq.codes.resize((size_t)n * PQ_M);
    f.read((char*)pq.pqcent.data(), sizeof(float) * pq.pqcent.size());
    f.read((char*)pq.codes.data(), pq.codes.size());
    return (bool)f;
}
static void save_pq_cache(const PQCodebook& pq, const char* path, int n)
{
    std::ofstream f(path, std::ios::binary);
    int hm = PQ_M, hk = PQ_K, hd = PQ_DSUB, hn = n;
    f.write((char*)&hm, 4); f.write((char*)&hk, 4); f.write((char*)&hd, 4); f.write((char*)&hn, 4);
    f.write((char*)pq.pqcent.data(), sizeof(float) * pq.pqcent.size());
    f.write((char*)pq.codes.data(), pq.codes.size());
}

// ── coarse cluster selection (same as exp_ivf_grouped.cu) ────────────────────
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

// ── exact warp-scan fine kernel (baseline for comparison, from grouped exp) ──
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

// ── IVF-PQ CORE 1+2: fused residual LUT + warp-per-slot ADC scan ─────────────
// One warp owns one (query, probe) slot.  It first builds the slot's residual
// LUT directly in shared memory (lut[m][j] = <(q - cent_c)^(m), pqcent[m][j]>,
// 2048 entries / 32 lanes, dsub=12 dot each) -- no giant global LUT tensor is
// ever materialized.  Then approx_ip = <q, cent_c> (bias, free from the coarse
// score matrix) + sum_m lut[m][code_m]: each lane scans a strided slice of the
// cluster's 8-byte codes keeping its own top-L in registers, L = ceil(p/32).
// The slot buffer stores all 32*L lane entries: their union always contains
// the slot's true top-L, and is the slot's EXACT candidate set whenever the
// cluster has <= 32*L vectors; otherwise it is the standard lane-local
// relaxation of the global-heap semantics.
__global__ void ivfpq_scan_kernel(const float* __restrict__ Q, const float* __restrict__ cent,
                                  const float* __restrict__ pqcent,
                                  const uint8_t* __restrict__ codes,
                                  const float* __restrict__ coarseS, int nlist,
                                  const int* __restrict__ coff, const int* __restrict__ cids,
                                  int qn, int nprobe, int laneL,
                                  float* __restrict__ cand_val, int* __restrict__ cand_pos)
{
    // block = 4 warps; per warp: 8 KB LUT + 384 B residual in shared memory
    __shared__ float sLUT[4][PQ_M * PQ_K];
    __shared__ float sQr[4][DIM];
    int warp_in_blk = threadIdx.x >> 5, lane = threadIdx.x & 31;
    long long slot = (long long)blockIdx.x * 4 + warp_in_blk;
    if (slot >= (long long)qn * nprobe) return;   // whole warp exits together
    int i = (int)(slot / nprobe);
    int c = cids[slot];
    float* qr = sQr[warp_in_blk];
    for (int t = lane; t < DIM; t += 32)
        qr[t] = Q[(size_t)i * DIM + t] - cent[(size_t)c * DIM + t];
    __syncwarp();
    float* s = sLUT[warp_in_blk];
    for (int e = lane; e < PQ_M * PQ_K; e += 32) {   // build LUT in shared mem
        int m = e >> 8, j = e & (PQ_K - 1);
        const float* pc = pqcent + (size_t)e * PQ_DSUB;
        const float* q  = qr + m * PQ_DSUB;
        float acc = 0.0f;
        #pragma unroll
        for (int t = 0; t < PQ_DSUB; ++t) acc += q[t] * pc[t];
        s[e] = acc;
    }
    __syncwarp();
    float bias = coarseS[(size_t)i * nlist + c];
    int lo = coff[c], hi = coff[c + 1];
    float bv[MAX_LANE_L]; int bp[MAX_LANE_L];        // per-lane top-L
    for (int t = 0; t < laneL; ++t) { bv[t] = -FLT_MAX; bp[t] = -1; }
    for (int pos = lo + lane; pos < hi; pos += 32) {
        const uint8_t* code = codes + (size_t)pos * PQ_M;
        float acc = bias;
        #pragma unroll
        for (int m = 0; m < PQ_M; ++m) acc += s[m * PQ_K + code[m]];
        if (acc > bv[laneL - 1]) {
            int pp = laneL - 1;
            while (pp > 0 && bv[pp - 1] < acc) { bv[pp] = bv[pp - 1]; bp[pp] = bp[pp - 1]; --pp; }
            bv[pp] = acc; bp[pp] = pos;
        }
    }
    size_t off = slot * (size_t)(32 * laneL) + (size_t)lane * laneL;
    for (int t = 0; t < laneL; ++t) { cand_val[off + t] = bv[t]; cand_pos[off + t] = bp[t]; }
}

// ── IVF-PQ CORE 3: warp-per-query merge of slot candidates -> rerank set ─────
// A query's candidates are contiguous (its nprobe slots are adjacent).  Each
// lane strides over them keeping a register top-L; the union of the 32 lane
// lists (exactly 32*L = p entries) is the rerank set.  Clusters are disjoint,
// so there are no duplicate candidates.
__global__ void ivfpq_merge_kernel(const float* __restrict__ cand_val,
                                   const int* __restrict__ cand_pos,
                                   int qn, int nprobe, int slot_stride, int laneL,
                                   int* __restrict__ topp_pos)
{
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = gtid >> 5, lane = gtid & 31;
    if (i >= qn) return;
    int total = nprobe * slot_stride;
    size_t base_off = (size_t)i * total;
    float bv[MAX_LANE_L]; int bp[MAX_LANE_L];
    for (int t = 0; t < laneL; ++t) { bv[t] = -FLT_MAX; bp[t] = -1; }
    for (int t = lane; t < total; t += 32) {
        float v = cand_val[base_off + t];
        if (v > bv[laneL - 1]) {
            int pp = laneL - 1;
            while (pp > 0 && bv[pp - 1] < v) { bv[pp] = bv[pp - 1]; bp[pp] = bp[pp - 1]; --pp; }
            bv[pp] = v; bp[pp] = cand_pos[base_off + t];
        }
    }
    size_t out = (size_t)i * (32 * laneL) + (size_t)lane * laneL;
    for (int t = 0; t < laneL; ++t) topp_pos[out + t] = bp[t];
}

// ── IVF-PQ CORE 4: exact rerank of the top-p on original vectors ─────────────
__global__ void ivfpq_rerank_kernel(const float* __restrict__ Q, const float* __restrict__ RB,
                                    const int* __restrict__ idmap,
                                    const int* __restrict__ topp_pos, int qn, int p,
                                    int32_t* __restrict__ out_ids)
{
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = gtid >> 5, lane = gtid & 31;
    if (i >= qn) return;
    const float* q = Q + (size_t)i * DIM;
    float q0 = q[lane], q1 = q[lane + 32], q2 = q[lane + 64];
    float bestVal[K]; int bestId[K];
    #pragma unroll
    for (int t = 0; t < K; ++t) { bestVal[t] = -FLT_MAX; bestId[t] = -1; }
    for (int t = 0; t < p; ++t) {
        int pos = topp_pos[(size_t)i * p + t];
        if (pos < 0) continue;
        const float* b = RB + (size_t)pos * DIM;
        float partial = q0 * b[lane] + q1 * b[lane + 32] + q2 * b[lane + 64];
        float ip = warpReduceSum(partial);
        if (lane == 0 && ip > bestVal[K - 1]) {
            int pp = K - 1;
            while (pp > 0 && bestVal[pp - 1] < ip) { bestVal[pp] = bestVal[pp - 1]; bestId[pp] = bestId[pp - 1]; --pp; }
            bestVal[pp] = ip; bestId[pp] = idmap[pos];
        }
    }
    if (lane == 0)
        #pragma unroll
        for (int t = 0; t < K; ++t) out_ids[(size_t)i * K + t] = bestId[t];
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
    printf("[mem ] fine-phase reads: fp32 base %.1f MB  vs  PQ codes %.1f MB (%.0fx)\n",
           n * DIM * 4.0 / 1048576.0, n * (double)PQ_M / 1048576.0,
           (DIM * 4.0) / PQ_M);

    // device buffers (sized for worst case in the sweep)
    const int SLOT_STRIDE_MAX = 32 * ((MAX_P + 31) / 32);          // 128
    float *d_query, *d_cent, *d_RB, *d_coarse, *d_pqcent, *d_lut, *d_cval, *d_tpv, *d_val;
    int   *d_coff, *d_idmap, *d_cids, *d_cpos, *d_tpp;
    uint8_t *d_codes;
    int32_t *d_topid;
    CUDA_CHECK(cudaMalloc(&d_query,  sizeof(float) * m * DIM));
    CUDA_CHECK(cudaMalloc(&d_cent,   sizeof(float) * (size_t)MAX_NLIST * DIM));
    CUDA_CHECK(cudaMalloc(&d_RB,     sizeof(float) * n * DIM));
    CUDA_CHECK(cudaMalloc(&d_coarse, sizeof(float) * (size_t)MAX_NLIST * QCHUNK));
    CUDA_CHECK(cudaMalloc(&d_pqcent, sizeof(float) * PQ_M * PQ_K * PQ_DSUB));
    CUDA_CHECK(cudaMalloc(&d_lut,    sizeof(float) * (size_t)QCHUNK * MAXNPROBE * PQ_M * PQ_K));
    CUDA_CHECK(cudaMalloc(&d_cval,   sizeof(float) * (size_t)QCHUNK * MAXNPROBE * SLOT_STRIDE_MAX));
    CUDA_CHECK(cudaMalloc(&d_cpos,   sizeof(int)   * (size_t)QCHUNK * MAXNPROBE * SLOT_STRIDE_MAX));
    CUDA_CHECK(cudaMalloc(&d_tpv,    sizeof(float) * (size_t)QCHUNK * MAX_P));
    CUDA_CHECK(cudaMalloc(&d_tpp,    sizeof(int)   * (size_t)QCHUNK * MAX_P));
    CUDA_CHECK(cudaMalloc(&d_val,    sizeof(float) * m * K));
    CUDA_CHECK(cudaMalloc(&d_coff,   sizeof(int) * (MAX_NLIST + 1)));
    CUDA_CHECK(cudaMalloc(&d_idmap,  sizeof(int) * n));
    CUDA_CHECK(cudaMalloc(&d_cids,   sizeof(int) * (size_t)QCHUNK * MAXNPROBE));
    CUDA_CHECK(cudaMalloc(&d_codes,  (size_t)n * PQ_M));
    CUDA_CHECK(cudaMalloc(&d_topid,  sizeof(int32_t) * m * K));
    CUDA_CHECK(cudaMemcpy(d_query, query, sizeof(float) * m * DIM, cudaMemcpyHostToDevice));

    cublasHandle_t blas; cublasCreate(&blas);
    std::vector<int32_t> ids(m * K);
    const float alpha = 1.0f, beta = 0.0f;
    const int REP = 3;

    FILE* csv = fopen("results_ivfpq.csv", "w");
    const char* header = "method,nlist,nprobe,p,recall,us_per_query,"
                         "coarse_ms,lut_ms,scan_ms,merge_ms,rerank_ms";
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

        PQCodebook pq;
        char pqcache[128]; snprintf(pqcache, sizeof(pqcache), "data/ivfpq_nlist%d_M%d.cache", nlist, PQ_M);
        if (!load_pq_cache(pq, pqcache, (int)n)) {
            printf("[pq ] nlist=%d training residual PQ (M=%d K=%d dsub=%d) ...\n", nlist, PQ_M, PQ_K, PQ_DSUB);
            double t0 = omp_get_wtime();
            build_residual_pq(pq, idx);
            printf("[pq ] trained in %.1f s\n", omp_get_wtime() - t0);
            save_pq_cache(pq, pqcache, (int)n);
        } else printf("[pq ] nlist=%d cache\n", nlist);

        CUDA_CHECK(cudaMemcpy(d_cent, idx.centroids.data(), sizeof(float) * (size_t)nlist * DIM, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_RB, idx.reordered_base.data(), sizeof(float) * n * DIM, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_coff, idx.cluster_off.data(), sizeof(int) * (nlist + 1), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_idmap, idx.id_map.data(), sizeof(int) * n, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pqcent, pq.pqcent.data(), sizeof(float) * pq.pqcent.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_codes, pq.codes.data(), pq.codes.size(), cudaMemcpyHostToDevice));

        for (int pi = 0; pi < (int)(sizeof(NPROBE_SWEEP) / sizeof(int)); ++pi) {
            int np = NPROBE_SWEEP[pi];
            if (np > nlist) continue;

            // ── exact warp-scan IVF baseline at same nlist/nprobe ────────────
            {
                float best = 1e30f;
                for (int r = 0; r < REP; ++r) {
                    CudaTimer tt; tt.start();
                    for (size_t q0i = 0; q0i < m; q0i += QCHUNK) {
                        int qn = (int)std::min((size_t)QCHUNK, m - q0i);
                        cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, nlist, qn, DIM,
                                    &alpha, d_cent, DIM, d_query + q0i * DIM, DIM, &beta, d_coarse, nlist);
                        select_topn_kernel<<<(qn + TPB - 1) / TPB, TPB>>>(d_coarse, qn, nlist, np, d_cids);
                        ivf_fine_kernel_warp<<<(qn * 32 + TPB - 1) / TPB, TPB>>>(
                            d_query + q0i * DIM, d_RB, d_coff, d_idmap, d_cids, qn, np,
                            d_topid + q0i * K, d_val + q0i * K);
                    }
                    CUDA_CHECK(cudaMemcpy(ids.data(), d_topid, sizeof(int32_t) * m * K, cudaMemcpyDeviceToHost));
                    best = std::min(best, tt.stop());
                }
                double rec = compute_recall(ids.data(), gt, gt_d, m, K);
                char line[320];
                snprintf(line, sizeof(line), "ivf_warp,%d,%d,0,%.5f,%.3f,0,0,0,0,0",
                         nlist, np, rec, best * 1000.0 / m);
                printf("%s\n", line); fprintf(csv, "%s\n", line); fflush(csv);
            }

            // ── IVF-PQ at each rerank depth p ────────────────────────────────
            for (int ppi = 0; ppi < (int)(sizeof(P_SWEEP) / sizeof(int)); ++ppi) {
                int p = P_SWEEP[ppi];
                int laneL = (p + 31) / 32;         // per-lane kept candidates
                int slot_stride = 32 * laneL;      // slot buffer entries (see scan kernel)
                float best = 1e30f;
                float ph_coarse = 0, ph_lut = 0, ph_scan = 0, ph_merge = 0, ph_rr = 0;
                for (int r = 0; r < REP; ++r) {
                    float c_coarse = 0, c_lut = 0, c_scan = 0, c_merge = 0, c_rr = 0;
                    CudaTimer tt; tt.start();
                    for (size_t q0i = 0; q0i < m; q0i += QCHUNK) {
                        int qn = (int)std::min((size_t)QCHUNK, m - q0i);
                        long long nslot = (long long)qn * np;
                        CudaTimer pt;
                        pt.start();
                        cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, nlist, qn, DIM,
                                    &alpha, d_cent, DIM, d_query + q0i * DIM, DIM, &beta, d_coarse, nlist);
                        select_topn_kernel<<<(qn + TPB - 1) / TPB, TPB>>>(d_coarse, qn, nlist, np, d_cids);
                        c_coarse += pt.stop();
                        pt.start();
                        long long lut_tot = nslot * PQ_M * PQ_K;
                        ivfpq_lut_kernel<<<(unsigned)((lut_tot + TPB - 1) / TPB), TPB>>>(
                            d_query + q0i * DIM, d_cent, d_pqcent, d_cids, qn, np, d_lut);
                        c_lut += pt.stop();
                        pt.start();
                        ivfpq_scan_kernel<<<(unsigned)((nslot + 3) / 4), TPB>>>(
                            d_codes, d_lut, d_coarse, nlist, d_coff, d_cids, qn, np, laneL, d_cval, d_cpos);
                        c_scan += pt.stop();
                        pt.start();
                        ivfpq_merge_kernel<<<(qn + TPB - 1) / TPB, TPB>>>(
                            d_cval, d_cpos, qn, np, slot_stride, p, d_tpv, d_tpp);
                        c_merge += pt.stop();
                        pt.start();
                        ivfpq_rerank_kernel<<<(qn * 32 + TPB - 1) / TPB, TPB>>>(
                            d_query + q0i * DIM, d_RB, d_idmap, d_tpp, qn, p, d_topid + q0i * K);
                        c_rr += pt.stop();
                    }
                    CUDA_CHECK(cudaMemcpy(ids.data(), d_topid, sizeof(int32_t) * m * K, cudaMemcpyDeviceToHost));
                    float t = tt.stop();
                    if (t < best) { best = t; ph_coarse = c_coarse; ph_lut = c_lut;
                                    ph_scan = c_scan; ph_merge = c_merge; ph_rr = c_rr; }
                }
                double rec = compute_recall(ids.data(), gt, gt_d, m, K);
                char line[320];
                snprintf(line, sizeof(line), "ivfpq,%d,%d,%d,%.5f,%.3f,%.2f,%.2f,%.2f,%.2f,%.2f",
                         nlist, np, p, rec, best * 1000.0 / m,
                         ph_coarse, ph_lut, ph_scan, ph_merge, ph_rr);
                printf("%s\n", line); fprintf(csv, "%s\n", line); fflush(csv);
            }
        }
    }
    fclose(csv);
    printf("\n[done] raw data written to results_ivfpq.csv\n");

    cublasDestroy(blas);
    cudaFree(d_query); cudaFree(d_cent); cudaFree(d_RB); cudaFree(d_coarse);
    cudaFree(d_pqcent); cudaFree(d_lut); cudaFree(d_cval); cudaFree(d_cpos);
    cudaFree(d_tpv); cudaFree(d_tpp); cudaFree(d_val); cudaFree(d_coff);
    cudaFree(d_idmap); cudaFree(d_cids); cudaFree(d_codes); cudaFree(d_topid);
    delete[] base; delete[] query; delete[] gt;
    return 0;
}
