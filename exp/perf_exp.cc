// =============================================================================
// exp/perf_exp.cc — Hardware counter profiling for ANN search algorithms
//
// Measures three counter groups for the query loop:
//   1. IPC        : cycles + instructions  →  instructions-per-cycle
//   2. Cache      : LLC load misses  →  MPKI (misses per kilo-instruction)
//   3. Branch     : branch mispredictions  →  miss rate %
//
// Select algorithm via -DSEARCH_ALG=N (see list below).
// Compile from project root:
//   g++ exp/perf_exp.cc -o exp/perf_exp -O2 -std=c++11 -fopenmp -lpthread \
//       -I. -DSEARCH_ALG=6
//
// Run:
//   ./exp/perf_exp
//
// If /proc/sys/kernel/perf_event_paranoid > 2, hardware counters are blocked
// and only wall-clock latency is reported.
// =============================================================================

// --- Algorithm constants (mirrors main.cc) -----------------------------------
#define FLAT_SCALAR       1
#define FLAT_NORMAL       2
#define FLAT_SIMD         3
#define FLAT_SIMD_UNROLL  4
#define SQ_NORMAL         5
#define SQ_SIMD           6
#define SQ_SDC            7
#define PQ_NORMAL         8
#define PQ_RERANK         9
#define PQ_FLAT_SIMD     10
#define PQ_CC_SIMD       11
#define PQ_CC_UNROLL     12
#define PQ_GATHER        13

#define PQ_BUILD_SCALAR   1
#define PQ_BUILD_SIMD     2

#ifndef SEARCH_ALG
#  define SEARCH_ALG  SQ_SIMD
#endif
#ifndef BUILD_PQ
#  define BUILD_PQ    PQ_BUILD_SCALAR
#endif
#ifndef COARSE_P
#  define COARSE_P    200
#endif

// --- System / perf includes --------------------------------------------------
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <vector>
#include <queue>
#include <utility>
#include <set>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <sys/time.h>

#include "ARM/Alg_normal/flat_scan.h"
#include "ARM/Alg_normal/flat_scan_normal.h"
#include "ARM/Alg_normal/sq_flat_normal.h"
#include "ARM/Alg_normal/pq_flat_normal.h"
#include "ARM/Alg_parallel/flat_simd.h"
#include "ARM/Alg_parallel/sq_flat_simd.h"
#include "ARM/Alg_parallel/pq_flat_simd.h"

static const size_t p = COARSE_P;

// --- Helpers -----------------------------------------------------------------

template<typename T>
static T* LoadData(const std::string& path, size_t& n, size_t& d)
{
    std::ifstream fin(path, std::ios::binary);
    fin.read((char*)&n, 4);
    fin.read((char*)&d, 4);
    T* data = new T[n * d];
    int sz = sizeof(T);
    for (size_t i = 0; i < n; ++i)
        fin.read((char*)data + i * d * sz, d * sz);
    std::cerr << "loaded " << path << "  n=" << n << "  d=" << d << "\n";
    return data;
}

static inline int64_t tv_diff_us(const struct timeval& a, const struct timeval& b)
{
    return (b.tv_sec * 1000000LL + b.tv_usec) - (a.tv_sec * 1000000LL + a.tv_usec);
}

// --- perf_event_open wrappers ------------------------------------------------

static int perf_fd(uint32_t type, uint64_t config)
{
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.type           = type;
    pe.size           = sizeof(pe);
    pe.config         = config;
    pe.disabled       = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv     = 1;
    return (int)syscall(SYS_perf_event_open, &pe, 0, -1, -1, 0);
}

static void pstart(int fd) { ioctl(fd, PERF_EVENT_IOC_RESET, 0); ioctl(fd, PERF_EVENT_IOC_ENABLE, 0); }
static void pstop (int fd) { ioctl(fd, PERF_EVENT_IOC_DISABLE, 0); }
static uint64_t pread(int fd) { uint64_t v = 0; read(fd, &v, 8); return v; }

// LLC load-miss event (PERF_TYPE_HW_CACHE)
static int perf_llc_miss_fd()
{
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.type     = PERF_TYPE_HW_CACHE;
    pe.size     = sizeof(pe);
    pe.config   = (PERF_COUNT_HW_CACHE_LL)
                | (PERF_COUNT_HW_CACHE_OP_READ   << 8)
                | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
    pe.disabled       = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv     = 1;
    return (int)syscall(SYS_perf_event_open, &pe, 0, -1, -1, 0);
}

// =============================================================================
int main()
{
    static const char* alg_names[] = {
        "",
        "flat_search",                                   //  1
        "flat_search_normal",                            //  2
        "simd_flat_search",                              //  3
        "simd_flat_search_unroll",                       //  4
        "sq_flat_search_normal",                         //  5
        "sq_flat_search_simd",                           //  6
        "sq_flat_search_sdc",                            //  7
        "pq_flat_search_normal",                         //  8
        "pq_flat_search_rerank",                         //  9
        "pq_flat_search_rerank_flat_simd",               // 10
        "pq_flat_search_rerank_cross_centroid_simd",     // 11
        "pq_flat_search_rerank_cc_unroll",               // 12
    };

    std::cout << "SEARCH_ALG=" << SEARCH_ALG
              << "  " << alg_names[SEARCH_ALG]
              << "  p=" << p << "\n";

    // --- Load data -----------------------------------------------------------
    const std::string data_path = "/anndata/";
    size_t test_number = 0, base_number = 0, test_gt_d = 0, vecdim = 0;

    float* queries = LoadData<float>(data_path + "DEEP100K.query.fbin",
                                     test_number, vecdim);
    int*   gt      = LoadData<int>  (data_path + "DEEP100K.gt.query.100k.top100.bin",
                                     test_number, test_gt_d);
    float* base    = LoadData<float>(data_path + "DEEP100K.base.100k.fbin",
                                     base_number, vecdim);
    test_number = 2000;
    const size_t k = 10;

    // --- Build index (conditional on algorithm) ------------------------------
    struct timeval tb0, tb1;

#if SEARCH_ALG >= SQ_NORMAL && SEARCH_ALG <= SQ_SDC
    SQIndex sq_index;
    gettimeofday(&tb0, NULL);
    sq_index.build(base, base_number, vecdim);
    gettimeofday(&tb1, NULL);
    std::cerr << "[build] SQIndex: " << tv_diff_us(tb0, tb1) / 1000 << " ms\n";
#endif

#if SEARCH_ALG >= PQ_NORMAL
    PQIndex pq_index;
    gettimeofday(&tb0, NULL);
#if BUILD_PQ == PQ_BUILD_SIMD
    pq_build_index_simd_blocked(pq_index, base, base_number, vecdim);
#else
    pq_index.build(base, base_number, vecdim);
#endif
    gettimeofday(&tb1, NULL);
    std::cerr << "[build] PQIndex: " << tv_diff_us(tb0, tb1) / 1000 << " ms\n";

#if SEARCH_ALG == PQ_CC_SIMD || SEARCH_ALG == PQ_CC_UNROLL || SEARCH_ALG == PQ_GATHER
    PQIndexSIMD pq_simd(pq_index);
#endif
#endif

    // --- Open perf counters --------------------------------------------------
    int fd_cyc  = perf_fd(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
    int fd_ins  = perf_fd(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
    int fd_llc  = perf_llc_miss_fd();
    int fd_br   = perf_fd(PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES);

    bool perf_ok = (fd_cyc >= 0 && fd_ins >= 0 && fd_llc >= 0 && fd_br >= 0);
    if (!perf_ok) {
        std::cerr << "[perf] perf_event_open blocked — "
                     "check /proc/sys/kernel/perf_event_paranoid (need <= 2)\n"
                     "[perf] falling back to wall-clock timing only\n";
        // close any that did open
        if (fd_cyc >= 0) close(fd_cyc);
        if (fd_ins >= 0) close(fd_ins);
        if (fd_llc >= 0) close(fd_llc);
        if (fd_br  >= 0) close(fd_br);
    }

    // --- Query loop ----------------------------------------------------------
    if (perf_ok) { pstart(fd_cyc); pstart(fd_ins); pstart(fd_llc); pstart(fd_br); }

    struct timeval t0, t1;
    gettimeofday(&t0, NULL);

    float total_recall = 0.0f;
    for (int i = 0; i < (int)test_number; ++i)
    {
        const float* q = queries + i * vecdim;

#if   SEARCH_ALG == FLAT_SCALAR
        auto res = flat_search(base, q, base_number, vecdim, k);
#elif SEARCH_ALG == FLAT_NORMAL
        auto res = flat_search_normal(base, q, base_number, vecdim, k);
#elif SEARCH_ALG == FLAT_SIMD
        auto res = simd_flat_search(base, q, base_number, vecdim, k);
#elif SEARCH_ALG == FLAT_SIMD_UNROLL
        auto res = simd_flat_search_unroll(base, q, base_number, vecdim, k);
#elif SEARCH_ALG == SQ_NORMAL
        auto res = sq_flat_search_normal(sq_index, base, q, k, p);
#elif SEARCH_ALG == SQ_SIMD
        auto res = sq_flat_search_simd(sq_index, base, q, k, p);
#elif SEARCH_ALG == SQ_SDC
        auto res = sq_flat_search_sdc(sq_index, base, q, k, p);
#elif SEARCH_ALG == PQ_NORMAL
        auto res = pq_flat_search_normal(pq_index, q, k);
#elif SEARCH_ALG == PQ_RERANK
        auto res = pq_flat_search_rerank(pq_index, base, q, k, p);
#elif SEARCH_ALG == PQ_FLAT_SIMD
        auto res = pq_flat_search_rerank_flat_simd(pq_index, base, q, k, p);
#elif SEARCH_ALG == PQ_CC_SIMD
        auto res = pq_flat_search_rerank_cross_centroid_simd(pq_simd, base, q, k, p);
#elif SEARCH_ALG == PQ_CC_UNROLL
        auto res = pq_flat_search_rerank_cc_unroll(pq_simd, base, q, k, p);
#elif SEARCH_ALG == PQ_GATHER
        auto res = pq_flat_search_rerank_gather(pq_simd, base, q, k, p);
#else
        #error "Unknown SEARCH_ALG value (1-13)."
#endif

        std::set<uint32_t> gtset;
        for (int j = 0; j < (int)k; ++j)
            gtset.insert((uint32_t)gt[j + i * test_gt_d]);
        size_t acc = 0;
        while (!res.empty()) {
            if (gtset.count(res.top().second)) ++acc;
            res.pop();
        }
        total_recall += (float)acc / k;
    }

    gettimeofday(&t1, NULL);
    int64_t wall_us = tv_diff_us(t0, t1);

    if (perf_ok) { pstop(fd_cyc); pstop(fd_ins); pstop(fd_llc); pstop(fd_br); }

    // --- Read counters -------------------------------------------------------
    uint64_t cyc = 0, ins = 0, llc = 0, br = 0;
    if (perf_ok) {
        cyc = pread(fd_cyc);
        ins = pread(fd_ins);
        llc = pread(fd_llc);
        br  = pread(fd_br);
        close(fd_cyc); close(fd_ins); close(fd_llc); close(fd_br);
    }

    // --- Report --------------------------------------------------------------
    std::cout << std::fixed;
    std::cout << "\n";
    std::cout << std::setprecision(4)
              << "recall:       " << total_recall / test_number << "\n"
              << std::setprecision(2)
              << "avg latency:  " << (double)wall_us / test_number << " us\n";

    if (perf_ok) {
        double ipc         = (double)ins / cyc;
        double mpki        = (double)llc / (ins / 1000.0);
        double br_miss_pct = (double)br  / ins * 100.0;

        std::cout << "\n"
                  << std::setprecision(3)
                  << "counter              total              per-query\n"
                  << "───────────────────────────────────────────────────\n"
                  << "cycles         " << std::setw(16) << cyc
                                       << std::setw(16) << cyc / test_number  << "\n"
                  << "instructions   " << std::setw(16) << ins
                                       << std::setw(16) << ins / test_number  << "\n"
                  << "IPC            " << std::setprecision(3) << ipc          << "\n"
                  << "LLC misses     " << std::setw(16) << llc
                                       << std::setw(16) << llc / test_number  << "\n"
                  << "MPKI           " << std::setprecision(3) << mpki
                                       << "  (LLC misses per kilo-instr)\n"
                  << "branch misses  " << std::setw(16) << br
                                       << std::setw(16) << br  / test_number  << "\n"
                  << "branch miss%   " << std::setprecision(3) << br_miss_pct  << "%\n";
    }

    delete[] queries;
    delete[] gt;
    delete[] base;
    return 0;
}
