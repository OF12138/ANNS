// =============================================================================
// main.cc — Benchmark harness for Approximate Nearest Neighbor (ANN) search
//
// Purpose:
//   This is the ONLY file you should modify for the assignment.
//   It loads the DEEP100K dataset, runs a search function for each of 2000
//   query vectors, computes recall against ground-truth labels, and reports
//   average recall and per-query latency in microseconds.
//
// Task:
//   Replace (or augment) the `flat_search` call with a faster approximate
//   algorithm (e.g., HNSW, SIMD-accelerated flat scan, IVF, etc.) while
//   maintaining high recall (ideally ≥ 0.9).
//
// Constraints:
//   - Do NOT modify flat_scan.h.
//   - The search function's return type must remain:
//       std::priority_queue<std::pair<float, uint32_t>>
//   - Index files must be saved under files/ (PBS cluster path limit).
//   - Do not build the index during the timed query loop (causes jitter).
//
// Data files (on the cluster at /anndata/):
//   DEEP100K.base.100k.fbin        — 100,000 base vectors (float32, dim=96)
//   DEEP100K.query.fbin            — query vectors (float32, dim=96)
//   DEEP100K.gt.query.100k.top100.bin — ground-truth top-100 labels (int32)
//
// Binary file format (.fbin / .bin):
//   [4 bytes: n (number of vectors)] [4 bytes: d (dimension)]
//   [n × d × sizeof(T) bytes: row-major vector data]
// =============================================================================

// =============================================================================
//  EXPERIMENT CONFIGURATION — change SEARCH_ALG, BUILD_PQ, and COARSE_P,
//  then recompile.
//
//  SEARCH_ALG values:
//   ── Exact flat scan ──────────────────────────────────────────────────────
//    1  FLAT_SCALAR      flat_search()                scalar, int loop, exact
//    2  FLAT_NORMAL      flat_search_normal()         scalar, size_t loop, exact
//    3  FLAT_SIMD        simd_flat_search()           NEON float32x4 1-acc, exact
//    4  FLAT_SIMD_UNROLL simd_flat_search_unroll()    NEON float32x4 4-acc, exact
//   ── Scalar Quantization (2-phase, coarse p candidates → exact rerank) ───
//    5  SQ_NORMAL        sq_flat_search_normal()      scalar ADC coarse + scalar rerank
//    6  SQ_SIMD          sq_flat_search_simd()        NEON ADC coarse (uint8→float) + NEON rerank
//    7  SQ_SDC           sq_flat_search_sdc()         NEON SDC coarse (vmull_u8 integer) + NEON rerank
//   ── Product Quantization (2-phase, coarse p candidates → exact rerank) ──
//    8  PQ_NORMAL        pq_flat_search_normal()      scalar ADC scan, no rerank
//    9  PQ_RERANK        pq_flat_search_rerank()      scalar ADC scan + scalar float rerank
//   10  PQ_FLAT_SIMD     pq_flat_search_rerank_flat_simd()          flat-SIMD LUT (1 acc/centroid)
//   11  PQ_CC_SIMD       pq_flat_search_rerank_cross_centroid_simd() CC-SIMD LUT (0 reductions)
//   12  PQ_CC_UNROLL     pq_flat_search_rerank_cc_unroll()           CC-SIMD LUT + 4× unroll
//   13  PQ_GATHER        pq_flat_search_rerank_gather()              cc_unroll LUT + NEON gather scan
//
//  BUILD_PQ values (only used when SEARCH_ALG is 8–13; ignored otherwise):
//    1  PQ_BUILD_SCALAR   pq_index.build()                scalar k-means
//    2  PQ_BUILD_SIMD     pq_build_index_simd_blocked()   SIMD double-tiled (i_blk × k_blk)
//
//  COARSE_P — coarse candidate count for SQ/PQ rerank (SEARCH_ALG 5–12).
//    Ignored for SEARCH_ALG 1–4 (exact flat scan needs no coarse filter).
//    Larger COARSE_P → higher recall, higher latency.
//    Typical range: 100 (fast) – 1000 (high recall).
// =============================================================================
#define FLAT_SCALAR            1
#define FLAT_NORMAL            2
#define FLAT_SIMD              3
#define FLAT_SIMD_UNROLL       4
#define SQ_NORMAL              5
#define SQ_SIMD                6
#define SQ_SDC                 7
#define PQ_NORMAL              8
#define PQ_RERANK              9
#define PQ_FLAT_SIMD          10
#define PQ_CC_SIMD            11
#define PQ_CC_UNROLL          12
#define PQ_GATHER             13
//   ── Pthread flat scan ────────────────────────────────────────────────────
//   14  FLAT_SIMD_QUERY_PTHREAD  simd_flat_search_query_parallel()  all queries split across threads
//   15  FLAT_SIMD_BASE_PTHREAD   simd_flat_search_base_parallel()   one query, base split across threads
//   ── OpenMP flat scan ─────────────────────────────────────────────────────
//   16  FLAT_SIMD_QUERY_OMP      simd_flat_search_query_parallel_omp()  OpenMP query-level
//   17  FLAT_SIMD_BASE_OMP       simd_flat_search_base_parallel_omp()   OpenMP base-partition
#define FLAT_SIMD_QUERY_PTHREAD   14
#define FLAT_SIMD_BASE_PTHREAD    15
#define FLAT_SIMD_QUERY_OMP       16
#define FLAT_SIMD_BASE_OMP        17
//   ── PQ query-parallel LUT construction ───────────────────────────────────
//   18  PQ_GATHER_QUERY_PTHREAD  pq_batch_build_lut_pthread() + gather scan
//   19  PQ_GATHER_QUERY_OMP      pq_batch_build_lut_omp()    + gather scan
#define PQ_GATHER_QUERY_PTHREAD   18
#define PQ_GATHER_QUERY_OMP       19
//   ── PQ query-parallel scan+rerank ────────────────────────────────────────
//   20  PQ_SCAN_QUERY_PTHREAD  pq_batch_scan_rerank_pthread()  gather scan parallel, pthread
//   21  PQ_SCAN_QUERY_OMP      pq_batch_scan_rerank_omp()      gather scan parallel, OMP
#define PQ_SCAN_QUERY_PTHREAD     20
#define PQ_SCAN_QUERY_OMP         21
//   ── IVF-SIMD baseline ────────────────────────────────────────────────────
//   22  IVF_SIMD                 ivf_search_simd()               two-phase: coarse + fine NEON (single-thread)
//   ── IVF-SIMD cluster-partition parallel ──────────────────────────────────
//   23  IVF_SIMD_CLUSTER_PTHREAD ivf_search_simd_cluster_pthread() fine-scan split across Pthreads
//        IVF_FLATTEN 0 = round-robin cluster assignment (non-uniform load)
//        IVF_FLATTEN 1 = flatten vector list, split evenly (default, balanced)
#define IVF_SIMD                  22
#define IVF_SIMD_CLUSTER_PTHREAD  23

#define PQ_BUILD_SCALAR        1
#define PQ_BUILD_SIMD          2

// ── SET YOUR EXPERIMENT HERE ─────────────────────────────────────────────────
#define SEARCH_ALG   FLAT_SIMD_UNROLL
#define BUILD_PQ     PQ_BUILD_SIMD
#define COARSE_P     200
// Number of Pthread worker threads (used by SEARCH_ALG 14 and 15).
// IVF parameters (used by SEARCH_ALG 22–23):
//   IVF_NLIST   : number of clusters (64–4096; sqrt(100K) ≈ 316, use 256 or 512)
//   IVF_NPROBE  : clusters scanned per query (latency-recall knob; sweep 1–nlist)
//   IVF_REORDER : 0 = original layout (random access), 1 = cluster-contiguous layout
//   IVF_FLATTEN : (SEARCH_ALG 23 only) 0 = cluster-split, 1 = flatten-then-split
#define IVF_NLIST    1024
#define IVF_NPROBE   16
#define IVF_REORDER  0
#define IVF_FLATTEN  1
// Server has 8 cores; 7 workers + 1 main thread = full utilisation.
#define FLAT_PTHREAD_THREADS   7
// ─────────────────────────────────────────────────────────────────────────────

// =============================================================================

#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include <omp.h>
#include "hnswlib/hnswlib/hnswlib.h"
#include "ARM/Alg_normal/flat_scan.h"
#include "ARM/Alg_normal/flat_scan_normal.h"
#include "ARM/Alg_normal/sq_flat_normal.h"
#include "ARM/Alg_normal/pq_flat_normal.h"
#include "ARM/Alg_parallel/flat_simd.h"
#include "ARM/Alg_parallel/flat_simd_pthread.h"
#include "ARM/Alg_parallel/flat_simd_omp.h"
#include "ARM/Alg_parallel/sq_flat_simd.h"
#include "ARM/Alg_parallel/pq_flat_simd.h"
#include "ARM/Alg_parallel/pq_flat_simd_lut_parallel.h"
#include "ARM/Alg_parallel/pq_flat_simd_scan_parallel.h"
#include "ARM/Alg_parallel/ivf_flat_simd.h"
#include "ARM/Alg_parallel/ivf_flat_simd_parallel.h"

using namespace hnswlib;

static const size_t p = COARSE_P;

// LoadData<T> — reads a binary vector file into a flat array
//
// File format: 4-byte n, 4-byte d, then n*d values of type T.
// Returns a heap-allocated array of size n*d; caller owns the memory.
// Sets n and d as output parameters.
template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i)
        fin.read(((char*)data + i*d*sz), d*sz);
    fin.close();

    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";

    return data;
}

// SearchResult — stores per-query benchmark metrics
struct SearchResult
{
    float recall;
    int64_t latency; // microseconds
};

// build_index — constructs and persists an HNSW index (example, disabled by default)
void build_index(float* base, size_t base_number, size_t vecdim)
{
    const int efConstruction = 150;
    const int M = 16;

    HierarchicalNSW<float> *appr_alg;
    InnerProductSpace ipspace(vecdim);
    appr_alg = new HierarchicalNSW<float>(&ipspace, base_number, M, efConstruction);

    appr_alg->addPoint(base, 0);
    #pragma omp parallel for
    for(int i = 1; i < base_number; ++i)
        appr_alg->addPoint(base + 1ll*vecdim*i, i);

    char path_index[1024] = "files/hnsw.index";
    appr_alg->saveIndex(path_index);
}

// Returns elapsed microseconds between two gettimeofday snapshots.
static inline int64_t tv_diff_us(const struct timeval& a, const struct timeval& b)
{
    return (b.tv_sec * 1000000LL + b.tv_usec) - (a.tv_sec * 1000000LL + a.tv_usec);
}

int main(int argc, char *argv[])
{
    // ── Print experiment configuration ───────────────────────────────────────
    // These names mirror the comment table above; shown in PBS stdout for easy
    // identification when comparing multiple job outputs.
    static const char* search_names[] = 
    {
        "",
        "flat_search (scalar exact)",                                    //  1
        "flat_search_normal (scalar exact, size_t)",                     //  2
        "simd_flat_search (NEON 1-acc exact)",                           //  3
        "simd_flat_search_unroll (NEON 4-acc exact)",                    //  4
        "sq_flat_search_normal (SQ scalar ADC)",                         //  5
        "sq_flat_search_simd (SQ NEON ADC float)",                       //  6
        "sq_flat_search_sdc (SQ NEON SDC integer)",                      //  7
        "pq_flat_search_normal (PQ scalar ADC no-rerank)",               //  8
        "pq_flat_search_rerank (PQ scalar ADC + rerank)",                //  9
        "pq_flat_search_rerank_flat_simd (PQ flat-SIMD LUT)",            // 10
        "pq_flat_search_rerank_cross_centroid_simd (PQ CC-SIMD LUT)",    // 11
        "pq_flat_search_rerank_cc_unroll (PQ CC-SIMD 4x-unroll LUT)",   // 12
        "pq_flat_search_rerank_gather (PQ gather scan + cc_unroll LUT)", // 13
        "simd_flat_search_query_parallel (pthread, query-level)",        // 14
        "simd_flat_search_base_parallel  (pthread, base-partition)",     // 15
        "simd_flat_search_query_parallel_omp (openmp, query-level)",     // 16
        "simd_flat_search_base_parallel_omp  (openmp, base-partition)",  // 17
        "pq_batch_build_lut_pthread + gather scan (pthread, query-parallel LUT)", // 18
        "pq_batch_build_lut_omp    + gather scan (omp,    query-parallel LUT)",   // 19
        "pq_batch_scan_rerank_pthread (pthread, query-parallel scan+rerank)",     // 20
        "pq_batch_scan_rerank_omp     (omp,    query-parallel scan+rerank)",      // 21
        "ivf_search_simd (IVF coarse+fine, NEON IP)",                             // 22
        "ivf_search_simd_cluster_pthread (IVF cluster-partition parallel)",       // 23
    };
    static const char* build_names[] = {
        "",
        "pq_index.build (scalar k-means)",                               //  1
        "pq_build_index_simd_blocked (SIMD double-tiled i×k)",           //  2
    };

    std::cerr << "========================================\n";
    std::cerr << "[config] search_alg = " << SEARCH_ALG
              << "  " << search_names[SEARCH_ALG] << "\n";
#if (SEARCH_ALG >= PQ_NORMAL && SEARCH_ALG <= PQ_GATHER) \
 || SEARCH_ALG == PQ_GATHER_QUERY_PTHREAD || SEARCH_ALG == PQ_GATHER_QUERY_OMP \
 || SEARCH_ALG == PQ_SCAN_QUERY_PTHREAD   || SEARCH_ALG == PQ_SCAN_QUERY_OMP
    std::cerr << "[config] build_pq   = " << BUILD_PQ
              << "  " << build_names[BUILD_PQ] << "\n";
#endif
#if SEARCH_ALG == FLAT_SIMD_QUERY_PTHREAD || SEARCH_ALG == FLAT_SIMD_BASE_PTHREAD \
 || SEARCH_ALG == FLAT_SIMD_QUERY_OMP     || SEARCH_ALG == FLAT_SIMD_BASE_OMP    \
 || SEARCH_ALG == PQ_GATHER_QUERY_PTHREAD || SEARCH_ALG == PQ_GATHER_QUERY_OMP   \
 || SEARCH_ALG == PQ_SCAN_QUERY_PTHREAD   || SEARCH_ALG == PQ_SCAN_QUERY_OMP     \
 || SEARCH_ALG == IVF_SIMD_CLUSTER_PTHREAD
    std::cerr << "[config] threads = " << FLAT_PTHREAD_THREADS << "\n";
#endif
    std::cerr << "[config] p=" << p << "  k=10\n";
#if SEARCH_ALG == IVF_SIMD || SEARCH_ALG == IVF_SIMD_CLUSTER_PTHREAD
    std::cerr << "[config] ivf_nlist="   << IVF_NLIST
              << "  ivf_nprobe=" << IVF_NPROBE
              << "  ivf_reorder=" << IVF_REORDER << "\n";
#endif
#if SEARCH_ALG == IVF_SIMD_CLUSTER_PTHREAD
    std::cerr << "[config] ivf_flatten=" << IVF_FLATTEN << "\n";
#endif
    std::cerr << "========================================\n";

    // ── Load dataset ─────────────────────────────────────────────────────────
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "/anndata/";
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin",              test_number, vecdim);
    auto test_gt    = LoadData<int>  (data_path + "DEEP100K.gt.query.100k.top100.bin",test_number, test_gt_d);
    auto base       = LoadData<float>(data_path + "DEEP100K.base.100k.fbin",          base_number, vecdim);
    test_number = 2000;

    const size_t k = 10;

    std::vector<SearchResult> results(test_number);

    // ── Index build (only constructs what SEARCH_ALG requires) ───────────────
    // Build time is printed to stderr so it appears in the PBS job's stderr
    // file separately from the recall/latency output in stdout.
    struct timeval tb0, tb1;

#if SEARCH_ALG >= SQ_NORMAL && SEARCH_ALG <= SQ_SDC
    SQIndex sq_index;
    gettimeofday(&tb0, NULL);
    sq_index.build(base, base_number, vecdim);
    gettimeofday(&tb1, NULL);
    std::cerr << "[build] SQIndex: " << tv_diff_us(tb0, tb1) / 1000 << " ms\n";
#endif

#if (SEARCH_ALG >= PQ_NORMAL && SEARCH_ALG <= PQ_GATHER) \
 || SEARCH_ALG == PQ_GATHER_QUERY_PTHREAD || SEARCH_ALG == PQ_GATHER_QUERY_OMP \
 || SEARCH_ALG == PQ_SCAN_QUERY_PTHREAD   || SEARCH_ALG == PQ_SCAN_QUERY_OMP
    PQIndex pq_index;
    gettimeofday(&tb0, NULL);
#if   BUILD_PQ == PQ_BUILD_SCALAR
    pq_index.build(base, base_number, vecdim);
#elif BUILD_PQ == PQ_BUILD_SIMD
    pq_build_index_simd_blocked(pq_index, base, base_number, vecdim);
#else
    #error "Unknown BUILD_PQ value. Use PQ_BUILD_SCALAR or PQ_BUILD_SIMD."
#endif
    gettimeofday(&tb1, NULL);
    std::cerr << "[build] PQIndex (build_pq=" << BUILD_PQ << "): "
              << tv_diff_us(tb0, tb1) / 1000 << " ms\n";

    // Transposed centroid layout — needed by CC-SIMD variants (11, 12, 13)
#if SEARCH_ALG == PQ_CC_SIMD || SEARCH_ALG == PQ_CC_UNROLL || SEARCH_ALG == PQ_GATHER \
 || SEARCH_ALG == PQ_GATHER_QUERY_PTHREAD || SEARCH_ALG == PQ_GATHER_QUERY_OMP  \
 || SEARCH_ALG == PQ_SCAN_QUERY_PTHREAD   || SEARCH_ALG == PQ_SCAN_QUERY_OMP
    PQIndexSIMD pq_simd(pq_index);
#endif
#endif

#if SEARCH_ALG == IVF_SIMD || SEARCH_ALG == IVF_SIMD_CLUSTER_PTHREAD
    IVFIndex ivf_index;
    {
        char ivf_cache[256];
        snprintf(ivf_cache, sizeof(ivf_cache),
                 "files/ivf_nlist%d_reorder%d.bin", IVF_NLIST, IVF_REORDER);
        gettimeofday(&tb0, NULL);
        if (ivf_load(ivf_index, ivf_cache)) {
            gettimeofday(&tb1, NULL);
            std::cerr << "[build] IVFIndex loaded from cache " << ivf_cache
                      << ": " << tv_diff_us(tb0, tb1) / 1000 << " ms\n";
        } else {
            ivf_build(ivf_index, base, base_number, vecdim,
                      IVF_NLIST, 25, IVF_REORDER != 0);
            gettimeofday(&tb1, NULL);
            std::cerr << "[build] IVFIndex built"
                      << " nlist=" << IVF_NLIST
                      << " reorder=" << IVF_REORDER
                      << ": " << tv_diff_us(tb0, tb1) / 1000 << " ms\n";
            if (ivf_save(ivf_index, ivf_cache))
                std::cerr << "[build] IVFIndex saved to " << ivf_cache << "\n";
            else
                std::cerr << "[build] WARNING: failed to save IVFIndex to " << ivf_cache << "\n";
        }
    }
#endif

    // ── LUT build phase timing ────────────────────────────────────────────────
    // Runs a LUT-build-only loop BEFORE the main timed loop.
    // The reported average latency below is NOT affected by this.
#if SEARCH_ALG >= PQ_RERANK && SEARCH_ALG <= PQ_GATHER
    double avg_lut_us = 0.0;
    {
        std::vector<float> dtable_tmp(pq_index.M * pq_index.K);
        double sum_lut = 0.0;
        for (int i = 0; i < (int)test_number; ++i) {
            const float* q = test_query + i * vecdim;
            struct timeval la, lb;
            gettimeofday(&la, NULL);
#if   SEARCH_ALG == PQ_RERANK
            for (size_t m = 0; m < pq_index.M; ++m) {
                const float* q_m    = q + m * pq_index.dsub;
                const float* c_base = pq_index.centroids.data() + m * pq_index.K * pq_index.dsub;
                for (size_t c = 0; c < pq_index.K; ++c) {
                    const float* cv = c_base + c * pq_index.dsub;
                    float ip = 0.0f;
                    for (size_t j = 0; j < pq_index.dsub; ++j) ip += q_m[j] * cv[j];
                    dtable_tmp[m * pq_index.K + c] = ip;
                }
            }
#elif SEARCH_ALG == PQ_FLAT_SIMD
            pq_build_lut_flat_simd(pq_index, q, dtable_tmp.data());
#elif SEARCH_ALG == PQ_CC_SIMD
            pq_build_lut_cross_centroid_simd(pq_simd, q, dtable_tmp.data());
#elif SEARCH_ALG == PQ_CC_UNROLL || SEARCH_ALG == PQ_GATHER
            pq_build_lut_cc_unroll(pq_simd, q, dtable_tmp.data());
#endif
            gettimeofday(&lb, NULL);
            sum_lut += tv_diff_us(la, lb);
        }
        avg_lut_us = sum_lut / test_number;
    }
    std::cerr << std::fixed << std::setprecision(2);
    std::cerr << "[phase] avg LUT build:   " << avg_lut_us << " us\n";
#endif

    // ── PQ query-parallel LUT: build all LUTs in parallel, time the LUT phase ──
    // all_dtables[i * M*K .. (i+1)*M*K) holds the pre-built LUT for query i.
    // The query loop (below) uses these LUTs directly for scan+rerank — the
    // per-iteration gettimeofday therefore measures scan+rerank only, not LUT.
    // Compare [parallel LUT] avg/query against [phase] avg LUT build (SEARCH_ALG 13)
    // to see the LUT speedup from parallelism.
#if SEARCH_ALG == PQ_GATHER_QUERY_PTHREAD || SEARCH_ALG == PQ_GATHER_QUERY_OMP
    const size_t lut_size = pq_simd.idx->M * pq_simd.idx->K;
    std::vector<float> all_dtables((size_t)test_number * lut_size);
    // warm-up run (fills caches, avoids cold-start bias)
    {
#if SEARCH_ALG == PQ_GATHER_QUERY_PTHREAD
        pq_batch_build_lut_pthread(pq_simd, test_query, (int)test_number, vecdim,
                                   all_dtables.data(), FLAT_PTHREAD_THREADS);
#else
        pq_batch_build_lut_omp(pq_simd, test_query, (int)test_number, vecdim,
                               all_dtables.data(), FLAT_PTHREAD_THREADS);
#endif
    }
    // measured run
    {
        struct timeval tl0, tl1;
        gettimeofday(&tl0, NULL);
#if SEARCH_ALG == PQ_GATHER_QUERY_PTHREAD
        pq_batch_build_lut_pthread(pq_simd, test_query, (int)test_number, vecdim,
                                   all_dtables.data(), FLAT_PTHREAD_THREADS);
#else
        pq_batch_build_lut_omp(pq_simd, test_query, (int)test_number, vecdim,
                               all_dtables.data(), FLAT_PTHREAD_THREADS);
#endif
        gettimeofday(&tl1, NULL);
        int64_t total_lut_us = tv_diff_us(tl0, tl1);
        std::cerr << std::fixed << std::setprecision(2);
        std::cerr << "[parallel LUT]"
                  << "  threads="    << FLAT_PTHREAD_THREADS
                  << "  total="      << total_lut_us << " us"
                  << "  avg/query="  << (double)total_lut_us / test_number << " us\n";
    }
#endif

    // ── PQ query-parallel scan+rerank: build LUTs once (serial), then parallel scan ──
    // LUT is built single-threaded so we isolate the scan phase speedup.
    // Compare [parallel scan] avg/query against [phase] avg scan+rerank (SEARCH_ALG 13).
#if SEARCH_ALG == PQ_SCAN_QUERY_PTHREAD || SEARCH_ALG == PQ_SCAN_QUERY_OMP
    using _PQResType = std::priority_queue<std::pair<float, uint32_t>>;
    const size_t scan_lut_size = pq_simd.idx->M * pq_simd.idx->K;

    // Build all LUTs sequentially (single-thread, same as baseline SEARCH_ALG 13)
    std::vector<float> scan_dtables((size_t)test_number * scan_lut_size);
    for (int i = 0; i < (int)test_number; ++i)
        pq_build_lut_cc_unroll(pq_simd, test_query + (size_t)i * vecdim,
                               scan_dtables.data() + (size_t)i * scan_lut_size);

    std::vector<_PQResType> pq_scan_results(test_number);

    // warm-up run
    {
#if SEARCH_ALG == PQ_SCAN_QUERY_PTHREAD
        pq_batch_scan_rerank_pthread(*pq_simd.idx, base, test_query, (int)test_number,
                                     vecdim, k, p, scan_dtables.data(),
                                     pq_scan_results.data(), FLAT_PTHREAD_THREADS);
#else
        pq_batch_scan_rerank_omp(*pq_simd.idx, base, test_query, (int)test_number,
                                  vecdim, k, p, scan_dtables.data(),
                                  pq_scan_results.data(), FLAT_PTHREAD_THREADS);
#endif
    }
    // measured run
    int64_t pq_scan_batch_avg_us = 0;
    {
        struct timeval ts0, ts1;
        gettimeofday(&ts0, NULL);
#if SEARCH_ALG == PQ_SCAN_QUERY_PTHREAD
        pq_batch_scan_rerank_pthread(*pq_simd.idx, base, test_query, (int)test_number,
                                     vecdim, k, p, scan_dtables.data(),
                                     pq_scan_results.data(), FLAT_PTHREAD_THREADS);
#else
        pq_batch_scan_rerank_omp(*pq_simd.idx, base, test_query, (int)test_number,
                                  vecdim, k, p, scan_dtables.data(),
                                  pq_scan_results.data(), FLAT_PTHREAD_THREADS);
#endif
        gettimeofday(&ts1, NULL);
        int64_t total_scan_us = tv_diff_us(ts0, ts1);
        pq_scan_batch_avg_us  = total_scan_us / (int64_t)test_number;
        std::cerr << std::fixed << std::setprecision(2);
        std::cerr << "[parallel scan]"
                  << "  threads="   << FLAT_PTHREAD_THREADS
                  << "  total="     << total_scan_us << " us"
                  << "  avg/query=" << (double)total_scan_us / test_number << " us\n";
    }
#endif

    // ── Pthread query-parallel: run all queries in one batch before the loop ──
    // Wall time is measured around the batch call; per-query latency is computed
    // as total_time / num_queries (throughput metric, not per-query latency).
#if SEARCH_ALG == FLAT_SIMD_QUERY_PTHREAD || SEARCH_ALG == FLAT_SIMD_QUERY_OMP
    using _PQType = std::priority_queue<std::pair<float, uint32_t>>;
    std::vector<_PQType> pth_batch(test_number);
    {
        struct timeval tq0, tq1;
        gettimeofday(&tq0, NULL);
#if SEARCH_ALG == FLAT_SIMD_QUERY_PTHREAD
        simd_flat_search_query_parallel(
            base, test_query, base_number, vecdim, k,
            (int)test_number, pth_batch.data(), FLAT_PTHREAD_THREADS);
#else
        simd_flat_search_query_parallel_omp(
            base, test_query, base_number, vecdim, k,
            (int)test_number, pth_batch.data(), FLAT_PTHREAD_THREADS);
#endif
        gettimeofday(&tq1, NULL);
        int64_t total_us = tv_diff_us(tq0, tq1);
        std::cerr << "[query-parallel warm-up]"
                  << "  threads=" << FLAT_PTHREAD_THREADS
                  << "  total="   << total_us << " us"
                  << "  avg/query=" << total_us / (int64_t)test_number << " us\n";
    }
    // pth_batch_avg will be used as the reported latency inside the loop below
    int64_t pth_batch_avg_us = 0;
    {
        struct timeval tq0, tq1;
        gettimeofday(&tq0, NULL);
#if SEARCH_ALG == FLAT_SIMD_QUERY_PTHREAD
        simd_flat_search_query_parallel(
            base, test_query, base_number, vecdim, k,
            (int)test_number, pth_batch.data(), FLAT_PTHREAD_THREADS);
#else
        simd_flat_search_query_parallel_omp(
            base, test_query, base_number, vecdim, k,
            (int)test_number, pth_batch.data(), FLAT_PTHREAD_THREADS);
#endif
        gettimeofday(&tq1, NULL);
        pth_batch_avg_us = tv_diff_us(tq0, tq1) / (int64_t)test_number;
    }
#endif

#if SEARCH_ALG == IVF_SIMD
    int64_t ivf_t_coarse = 0, ivf_t_scan = 0;
#endif
#if SEARCH_ALG == IVF_SIMD_CLUSTER_PTHREAD
    int64_t ivf_t_coarse = 0, ivf_t_scan = 0, ivf_t_merge = 0;
    int64_t ivf_t_create = 0, ivf_t_join  = 0;
#endif

    // ── Query loop ────────────────────────────────────────────────────────────
    for(int i = 0; i < (int)test_number; ++i)
    {
        struct timeval val, newVal;
        gettimeofday(&val, NULL);

#if   SEARCH_ALG == FLAT_SCALAR
        auto res = flat_search(base, test_query + i*vecdim, base_number, vecdim, k);
#elif SEARCH_ALG == FLAT_NORMAL
        auto res = flat_search_normal(base, test_query + i*vecdim, base_number, vecdim, k);
#elif SEARCH_ALG == FLAT_SIMD
        auto res = simd_flat_search(base, test_query + i*vecdim, base_number, vecdim, k);
#elif SEARCH_ALG == FLAT_SIMD_UNROLL
        auto res = simd_flat_search_unroll(base, test_query + i*vecdim, base_number, vecdim, k);
#elif SEARCH_ALG == SQ_NORMAL
        auto res = sq_flat_search_normal(sq_index, base, test_query + i*vecdim, k, p);
#elif SEARCH_ALG == SQ_SIMD
        auto res = sq_flat_search_simd(sq_index, base, test_query + i*vecdim, k, p);
#elif SEARCH_ALG == SQ_SDC
        auto res = sq_flat_search_sdc(sq_index, base, test_query + i*vecdim, k, p);
#elif SEARCH_ALG == PQ_NORMAL
        auto res = pq_flat_search_normal(pq_index, test_query + i*vecdim, k);
#elif SEARCH_ALG == PQ_RERANK
        auto res = pq_flat_search_rerank(pq_index, base, test_query + i*vecdim, k, p);
#elif SEARCH_ALG == PQ_FLAT_SIMD
        auto res = pq_flat_search_rerank_flat_simd(pq_index, base, test_query + i*vecdim, k, p);
#elif SEARCH_ALG == PQ_CC_SIMD
        auto res = pq_flat_search_rerank_cross_centroid_simd(pq_simd, base, test_query + i*vecdim, k, p);
#elif SEARCH_ALG == PQ_CC_UNROLL
        auto res = pq_flat_search_rerank_cc_unroll(pq_simd, base, test_query + i*vecdim, k, p);
#elif SEARCH_ALG == PQ_GATHER
        auto res = pq_flat_search_rerank_gather(pq_simd, base, test_query + i*vecdim, k, p);
#elif SEARCH_ALG == FLAT_SIMD_QUERY_PTHREAD
        // Results were computed in the pre-loop batch; just move them out.
        auto res = std::move(pth_batch[i]);
#elif SEARCH_ALG == FLAT_SIMD_BASE_PTHREAD
        auto res = simd_flat_search_base_parallel(
            base, test_query + i*vecdim, base_number, vecdim, k, FLAT_PTHREAD_THREADS);
#elif SEARCH_ALG == FLAT_SIMD_QUERY_OMP
        // Results were computed in the pre-loop batch; just move them out.
        auto res = std::move(pth_batch[i]);
#elif SEARCH_ALG == FLAT_SIMD_BASE_OMP
        auto res = simd_flat_search_base_parallel_omp(
            base, test_query + i*vecdim, base_number, vecdim, k, FLAT_PTHREAD_THREADS);
#elif SEARCH_ALG == PQ_GATHER_QUERY_PTHREAD || SEARCH_ALG == PQ_GATHER_QUERY_OMP
        // LUT was pre-built in parallel above; scan+rerank only here.
        auto res = pq_rerank_from_dtable_gather(
            *pq_simd.idx, base, test_query + i*vecdim, k, p,
            all_dtables.data() + (size_t)i * lut_size);
#elif SEARCH_ALG == PQ_SCAN_QUERY_PTHREAD || SEARCH_ALG == PQ_SCAN_QUERY_OMP
        // Results computed in parallel batch above; move out for recall eval.
        auto res = std::move(pq_scan_results[i]);
#elif SEARCH_ALG == IVF_SIMD
        auto res = ivf_search_simd_timed(ivf_index, base,
                                   test_query + i*vecdim, k, IVF_NPROBE,
                                   &ivf_t_coarse, &ivf_t_scan);
#elif SEARCH_ALG == IVF_SIMD_CLUSTER_PTHREAD
        auto res = ivf_search_simd_cluster_pthread_timed(ivf_index, base,
                                   test_query + i*vecdim, k, IVF_NPROBE,
                                   FLAT_PTHREAD_THREADS,
                                   &ivf_t_coarse, &ivf_t_scan, &ivf_t_merge,
                                   &ivf_t_create, &ivf_t_join);
#else
        #error "Unknown SEARCH_ALG value. Set it to one of the defined constants (1–23)."
#endif

        gettimeofday(&newVal, NULL);
        int64_t diff = tv_diff_us(val, newVal);
        // Query-parallel: override diff with the batch-level throughput latency.
        // The per-iteration gettimeofday only measures the std::move, not the search.
#if SEARCH_ALG == FLAT_SIMD_QUERY_PTHREAD || SEARCH_ALG == FLAT_SIMD_QUERY_OMP
        diff = pth_batch_avg_us;
#endif
#if SEARCH_ALG == PQ_SCAN_QUERY_PTHREAD || SEARCH_ALG == PQ_SCAN_QUERY_OMP
        diff = pq_scan_batch_avg_us;
#endif

        std::set<uint32_t> gtset;
        for(int j = 0; j < (int)k; ++j)
            gtset.insert(test_gt[j + i*test_gt_d]);

        size_t acc = 0;
        while (res.size())
        {
            if(gtset.count(res.top().second)) ++acc;
            res.pop();
        }
        results[i] = {(float)acc / k, diff};
    }

#if SEARCH_ALG == IVF_SIMD
    {
        double n = (double)test_number;
        double avg_coarse = ivf_t_coarse / n;
        double avg_scan   = ivf_t_scan   / n;
        double avg_total  = avg_coarse + avg_scan;
        std::cerr << "[phase] avg coarse (centroid rank): " << avg_coarse << " us  ("
                  << (avg_coarse / avg_total * 100.0) << "%)\n";
        std::cerr << "[phase] avg scan   (fine vectors):  " << avg_scan   << " us  ("
                  << (avg_scan   / avg_total * 100.0) << "%)\n";
        std::cerr << "[phase] avg total  (2 phases):      " << avg_total  << " us\n";
    }
#endif
#if SEARCH_ALG == IVF_SIMD_CLUSTER_PTHREAD
    {
        double n = (double)test_number;
        double avg_coarse  = ivf_t_coarse  / n;
        double avg_scan    = ivf_t_scan    / n;
        double avg_merge   = ivf_t_merge   / n;
        double avg_create  = ivf_t_create  / n;
        double avg_join    = ivf_t_join    / n;
        double avg_flatbld = avg_scan - avg_create - avg_join;
        double avg_total   = avg_coarse + avg_scan + avg_merge;
        std::cerr << "[phase] avg coarse  (centroid rank):  " << avg_coarse  << " us  ("
                  << (avg_coarse  / avg_total * 100.0) << "%)\n";
        std::cerr << "[phase] avg scan    (total Phase 2):  " << avg_scan    << " us  ("
                  << (avg_scan    / avg_total * 100.0) << "%)\n";
        std::cerr << "[phase]   flat-bld  (list build):     " << avg_flatbld << " us\n";
        std::cerr << "[phase]   create    (pthread_create):  " << avg_create  << " us\n";
        std::cerr << "[phase]   join      (parallel work):  " << avg_join    << " us\n";
        std::cerr << "[phase] avg merge   (heap merge):     " << avg_merge   << " us  ("
                  << (avg_merge   / avg_total * 100.0) << "%)\n";
        std::cerr << "[phase] avg total   (3 phases):       " << avg_total   << " us\n";
    }
#endif

    // ── Report ────────────────────────────────────────────────────────────────
    float avg_recall = 0, avg_latency = 0;
    for(int i = 0; i < (int)test_number; ++i)
    {
        avg_recall  += results[i].recall;
        avg_latency += results[i].latency;
    }
    // 浮点误差可能导致一些精确算法平均recall不是1
    std::cout << "average recall: "       << avg_recall  / test_number << "\n";
    std::cout << "average latency (us): " << avg_latency / test_number << "\n";
#if SEARCH_ALG >= PQ_RERANK && SEARCH_ALG <= PQ_GATHER
    {
        double avg_total      = (double)avg_latency / test_number;
        double scan_rerank_us = avg_total - avg_lut_us;
        std::cerr << "[phase] avg scan+rerank: " << scan_rerank_us << " us\n";
        std::cerr << "[phase] avg total:       " << avg_total      << " us\n";
        std::cerr << "[phase] lut%="
                  << (avg_lut_us       / avg_total * 100.0) << "  "
                  << "scan+rerank%="
                  << (scan_rerank_us   / avg_total * 100.0) << "\n";
    }
#endif
    return 0;
}
