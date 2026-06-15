// =============================================================================
// ivf_knn.cu -- Basic GPU IVF kNN + latency-vs-recall experiment vs brute NNS
//
// IVF = inverted file (coarse quantization).  Two phases per query:
//   COARSE: GEMM query x centroids (m x nlist), select the nprobe nearest
//           centroids per query.  Cheap (nlist << n).
//   FINE  : scan only the vectors inside those nprobe clusters, take top-k.
//           Far less work than brute force -> lower latency, at the cost of
//           recall (a true neighbor in an unprobed cluster is missed).
//
// nprobe is the latency<->recall knob.  This file sweeps it and also runs the
// brute-force cuBLAS NNS (recall ~1.0) as the reference point, writing a CSV of
// (method, nprobe, latency, recall) for an Excel latency-vs-recall chart.
//
// Index build = spherical k-means on CPU (OpenMP), cached to disk.  Base is
// reordered so each cluster is contiguous in memory (sequential fine scan).
//
//   >>> CORE CODE OF BASIC IVF (for the report) <<<
//     build : build_ivf()              -- k-means + cluster layout   ("IVF BUILD CORE")
//     coarse: ivf_search() cublasSgemm + select_topn_kernel          ("IVF COARSE CORE")
//     fine  : ivf_fine_kernel()                                       ("IVF FINE CORE")
//
// Build:  build.bat ivf_knn.cu ivf_knn.exe -Xcompiler /openmp -lcublas
// Run  :  ./ivf_knn
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
#define NLIST      1024     // IVF clusters
#define KM_ITERS   20       // k-means iterations
#define MAXNPROBE  64       // upper bound for per-thread cluster-id buffer


// ── IVF index (host side) ────────────────────────────────────────────────────
struct IVFIndex {
    int nlist = NLIST, d = DIM, n = 0;
    std::vector<float> centroids;       // nlist*d, unit-normalized
    std::vector<float> reordered_base;  // n*d, grouped by cluster
    std::vector<int>   cluster_off;     // nlist+1 prefix sums
    std::vector<int>   id_map;          // reordered pos -> original base id
};


// ── (1) IVF BUILD CORE: spherical k-means + contiguous cluster layout ────────
static void build_ivf(IVFIndex& idx, const float* base, int n, int d, int nlist)
{
    idx.n = n; idx.d = d; idx.nlist = nlist;
    idx.centroids.assign((size_t)nlist * d, 0.0f);
    std::vector<int> assign(n, 0);

    // init centroids = nlist distinct random base vectors (fixed seed)
    std::mt19937 rng(42);
    std::vector<int> perm(n);
    for (int i = 0; i < n; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    for (int c = 0; c < nlist; ++c)
        std::copy(base + (size_t)perm[c] * d, base + (size_t)perm[c] * d + d,
                  idx.centroids.begin() + (size_t)c * d);

    for (int it = 0; it < KM_ITERS; ++it) {
        // assignment step: nearest centroid by inner product (vectors are unit norm)
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
        // update step: centroid = normalized mean of its members
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
            if (cnt[c] == 0) {                                  // reseed dead cluster
                int r = perm[(c + it) % n];
                std::copy(base + (size_t)r * d, base + (size_t)r * d + d, ce);
                continue;
            }
            double nrm = 0.0;
            for (int t = 0; t < d; ++t) { double v = sum[(size_t)c * d + t] / cnt[c]; ce[t] = (float)v; nrm += v * v; }
            nrm = std::sqrt(nrm);
            if (nrm > 1e-12) for (int t = 0; t < d; ++t) ce[t] /= (float)nrm;
        }
    }

    // final assignment + contiguous cluster layout (counting sort)
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


// ── disk cache (centroids + offsets + id_map; reordered_base rebuilt) ────────
static bool load_ivf_cache(IVFIndex& idx, const char* path, const float* base)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;
    int nlist, d, n;
    f.read((char*)&nlist, 4); f.read((char*)&d, 4); f.read((char*)&n, 4);
    if (nlist != NLIST || d != DIM) return false;
    idx.nlist = nlist; idx.d = d; idx.n = n;
    idx.centroids.resize((size_t)nlist * d);
    idx.cluster_off.resize(nlist + 1);
    idx.id_map.resize(n);
    f.read((char*)idx.centroids.data(), sizeof(float) * (size_t)nlist * d);
    f.read((char*)idx.cluster_off.data(), sizeof(int) * (nlist + 1));
    f.read((char*)idx.id_map.data(), sizeof(int) * n);
    if (!f) return false;
    idx.reordered_base.resize((size_t)n * d);          // rebuild from base + id_map
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


// ── select nprobe nearest centroids per query (from coarse GEMM scores) ──────
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


// ── (3) IVF FINE CORE: scan the nprobe selected clusters, take top-k ─────────
__global__ void ivf_fine_kernel(const float* __restrict__ Q,
                                const float* __restrict__ RB,
                                const int* __restrict__ coff,
                                const int* __restrict__ idmap,
                                const int* __restrict__ cids,
                                int qn, int nprobe,
                                int32_t* __restrict__ out_ids,
                                float* __restrict__ out_val)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= qn) return;
    float bestVal[K]; int bestId[K];
    #pragma unroll
    for (int t = 0; t < K; ++t) { bestVal[t] = -FLT_MAX; bestId[t] = -1; }

    const float* q = Q + (size_t)i * DIM;
    for (int p = 0; p < nprobe; ++p) {
        int c  = cids[(size_t)i * nprobe + p];
        int lo = coff[c], hi = coff[c + 1];
        for (int pos = lo; pos < hi; ++pos) {
            const float* b = RB + (size_t)pos * DIM;
            float acc = 0.0f;
            #pragma unroll
            for (int t = 0; t < DIM; ++t) acc += q[t] * b[t];
            if (acc > bestVal[K - 1]) {
                int pp = K - 1;
                while (pp > 0 && bestVal[pp - 1] < acc) {
                    bestVal[pp] = bestVal[pp - 1]; bestId[pp] = bestId[pp - 1]; --pp;
                }
                bestVal[pp] = acc; bestId[pp] = idmap[pos];   // original base id
            }
        }
    }
    #pragma unroll
    for (int t = 0; t < K; ++t) {
        out_ids[(size_t)i * K + t] = bestId[t];
        out_val[(size_t)i * K + t] = bestVal[t];
    }
}


// ── brute-force NNS reference (cuBLAS distance + linear-scan top-k) ───────────
__global__ void nns_topk_kernel(const float* __restrict__ S, int qn, int n,
                                int32_t* __restrict__ out_ids, float* __restrict__ out_val)
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
            while (p > 0 && bestVal[p - 1] < v) { bestVal[p] = bestVal[p - 1]; bestId[p] = bestId[p - 1]; --p; }
            bestVal[p] = v; bestId[p] = j;
        }
    }
    #pragma unroll
    for (int t = 0; t < K; ++t) { out_ids[(size_t)i * K + t] = bestId[t]; out_val[(size_t)i * K + t] = bestVal[t]; }
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

    // ── build or load IVF ────────────────────────────────────────────────────
    IVFIndex idx;
    char cache[128]; snprintf(cache, sizeof(cache), "data/ivf_nlist%d.cache", NLIST);
    CudaTimer bt; bt.start();
    if (load_ivf_cache(idx, cache, base)) {
        printf("[ivf] loaded cache %s\n", cache);
    } else {
        printf("[ivf] building k-means nlist=%d iters=%d ...\n", NLIST, KM_ITERS);
        build_ivf(idx, base, (int)n, (int)DIM, NLIST);
        save_ivf_cache(idx, cache);
        printf("[ivf] built + cached (%.1f ms)\n", bt.stop());
    }

    // ── device upload (index resident; one-time, excluded from search latency)─
    float *d_base, *d_qchunk, *d_cent, *d_RB, *d_coarse, *d_S, *d_val;
    int   *d_coff, *d_idmap, *d_cids;
    int32_t *d_ids;
    CUDA_CHECK(cudaMalloc(&d_base,   sizeof(float) * n * DIM));
    CUDA_CHECK(cudaMalloc(&d_qchunk, sizeof(float) * QCHUNK * DIM));
    CUDA_CHECK(cudaMalloc(&d_cent,   sizeof(float) * (size_t)NLIST * DIM));
    CUDA_CHECK(cudaMalloc(&d_RB,     sizeof(float) * n * DIM));
    CUDA_CHECK(cudaMalloc(&d_coff,   sizeof(int) * (NLIST + 1)));
    CUDA_CHECK(cudaMalloc(&d_idmap,  sizeof(int) * n));
    CUDA_CHECK(cudaMalloc(&d_coarse, sizeof(float) * (size_t)QCHUNK * NLIST));
    CUDA_CHECK(cudaMalloc(&d_cids,   sizeof(int) * (size_t)QCHUNK * MAXNPROBE));
    CUDA_CHECK(cudaMalloc(&d_S,      sizeof(float) * (size_t)QCHUNK * n));  // NNS ref
    CUDA_CHECK(cudaMalloc(&d_ids,    sizeof(int32_t) * QCHUNK * K));
    CUDA_CHECK(cudaMalloc(&d_val,    sizeof(float) * QCHUNK * K));
    CUDA_CHECK(cudaMemcpy(d_base, base, sizeof(float) * n * DIM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cent, idx.centroids.data(), sizeof(float) * (size_t)NLIST * DIM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_RB, idx.reordered_base.data(), sizeof(float) * n * DIM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_coff, idx.cluster_off.data(), sizeof(int) * (NLIST + 1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_idmap, idx.id_map.data(), sizeof(int) * n, cudaMemcpyHostToDevice));

    cublasHandle_t blas; cublasCreate(&blas);
    std::vector<int32_t> ids(m * K);
    const float alpha = 1.0f, beta = 0.0f;

    // ── IVF search over all m queries (chunked); returns latency ms ──────────
    auto ivf_search = [&](int nprobe) -> float {
        CudaTimer tt; tt.start();
        for (size_t q0 = 0; q0 < m; q0 += QCHUNK) {
            int qn = (int)std::min((size_t)QCHUNK, m - q0);
            CUDA_CHECK(cudaMemcpy(d_qchunk, query + q0 * DIM, sizeof(float) * qn * DIM, cudaMemcpyHostToDevice));
            // ── IVF COARSE CORE: GEMM query x centroids 
            cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, NLIST, qn, DIM,
                        &alpha, d_cent, DIM, d_qchunk, DIM, &beta, d_coarse, NLIST);
            select_topn_kernel<<<(qn + TPB - 1) / TPB, TPB>>>(d_coarse, qn, NLIST, nprobe, d_cids);
            // ── fine scan ──
            ivf_fine_kernel<<<(qn + TPB - 1) / TPB, TPB>>>(
                d_qchunk, d_RB, d_coff, d_idmap, d_cids, qn, nprobe, d_ids, d_val);
            CUDA_CHECK(cudaMemcpy(ids.data() + q0 * K, d_ids, sizeof(int32_t) * qn * K, cudaMemcpyDeviceToHost));
        }
        return tt.stop();
    };

    // ── brute-force NNS reference over all m queries ─────────────────────────
    auto nns_search = [&]() -> float {
        CudaTimer tt; tt.start();
        for (size_t q0 = 0; q0 < m; q0 += QCHUNK) {
            int qn = (int)std::min((size_t)QCHUNK, m - q0);
            CUDA_CHECK(cudaMemcpy(d_qchunk, query + q0 * DIM, sizeof(float) * qn * DIM, cudaMemcpyHostToDevice));
            cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, (int)n, qn, DIM,
                        &alpha, d_base, DIM, d_qchunk, DIM, &beta, d_S, (int)n);
            nns_topk_kernel<<<(qn + TPB - 1) / TPB, TPB>>>(d_S, qn, (int)n, d_ids, d_val);
            CUDA_CHECK(cudaMemcpy(ids.data() + q0 * K, d_ids, sizeof(int32_t) * qn * K, cudaMemcpyDeviceToHost));
        }
        return tt.stop();
    };

    const int REP = 5;
    auto best = [&](auto fn) { float b = 1e30f; for (int r = 0; r < REP; ++r) b = std::min(b, fn()); return b; };

    FILE* csv = fopen("results_ivf_vs_nns.csv", "w");
    const char* header = "method,nprobe,latency_ms,us_per_query,recall";
    printf("\n%s\n", header); fprintf(csv, "%s\n", header);

    // NNS reference
    {
        float lat = best([&]{ return nns_search(); });
        double rec = compute_recall(ids.data(), gt, gt_d, m, K);
        char line[256]; snprintf(line, sizeof(line), "NNS-brute,%d,%.3f,%.3f,%.5f", (int)n, lat, 1000.0 * lat / m, rec);
        printf("%s\n", line); fprintf(csv, "%s\n", line);
    }
    // IVF sweep over nprobe
    const int sweep[] = {1, 2, 4, 8, 16, 32, 64};
    for (int s = 0; s < (int)(sizeof(sweep) / sizeof(int)); ++s) {
        int np = sweep[s];
        float lat = best([&]{ return ivf_search(np); });
        double rec = compute_recall(ids.data(), gt, gt_d, m, K);
        char line[256]; snprintf(line, sizeof(line), "IVF,%d,%.3f,%.3f,%.5f", np, lat, 1000.0 * lat / m, rec);
        printf("%s\n", line); fprintf(csv, "%s\n", line);
    }
    fclose(csv);
    printf("\n[done] raw data written to results_ivf_vs_nns.csv\n");

    cublasDestroy(blas);
    cudaFree(d_base); cudaFree(d_qchunk); cudaFree(d_cent); cudaFree(d_RB);
    cudaFree(d_coff); cudaFree(d_idmap); cudaFree(d_coarse); cudaFree(d_cids);
    cudaFree(d_S); cudaFree(d_ids); cudaFree(d_val);
    delete[] base; delete[] query; delete[] gt;
    return 0;
}

