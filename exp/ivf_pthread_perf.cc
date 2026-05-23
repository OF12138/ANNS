// =============================================================================
// exp/ivf_pthread_perf.cc — Per-thread hardware counter profiling for
//                           IVF_SIMD_CLUSTER_PTHREAD (SEARCH_ALG 23)
//
// Each Pthread worker opens its own perf_event FDs (pid=0 inside the thread =
// "this thread only"), so counters are isolated per thread.
//
// Reports per query (averaged over PROFILE_QUERIES queries):
//   • Per-thread: vectors scanned, cycles, instructions, LLC misses, MPKI
//   • Total:      phase timings (coarse / scan / merge) matching main.cc output
//
// Build (from project root):
//   g++ exp/ivf_pthread_perf.cc -o exp/ivf_pthread_perf -O2 -std=c++11 \
//       -fopenmp -lpthread -I. -DIVF_FLATTEN=0
//   g++ exp/ivf_pthread_perf.cc -o exp/ivf_pthread_perf -O2 -std=c++11 \
//       -fopenmp -lpthread -I. -DIVF_FLATTEN=1
//
// Run:
//   ./exp/ivf_pthread_perf [nprobe]     (default nprobe = IVF_NPROBE)
//
// Requires: /proc/sys/kernel/perf_event_paranoid <= 2
// =============================================================================

#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <unistd.h>
#include <pthread.h>

#include <vector>
#include <queue>
#include <utility>
#include <algorithm>
#include <set>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cstdio>

// ── IVF config ───────────────────────────────────────────────────────────────
#define IVF_NLIST    1024
#define IVF_NPROBE   16
#define IVF_REORDER  0
#ifndef IVF_FLATTEN
#  define IVF_FLATTEN 1
#endif
#define NUM_THREADS  7
#define PROFILE_QUERIES 500     // queries to average over (skip first as warmup)
#define WARMUP_QUERIES   50

// ── Algorithm headers ────────────────────────────────────────────────────────
#include "ARM/Alg_parallel/flat_simd.h"
#include "ARM/Alg_parallel/ivf_flat_simd.h"

// ── perf helpers ─────────────────────────────────────────────────────────────
// Called from inside a worker thread: pid=0 means "this thread", cpu=-1 = any.

static int perf_open_thread(uint32_t type, uint64_t config)
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

static int perf_llc_miss_thread()
{
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.type     = PERF_TYPE_HW_CACHE;
    pe.size     = sizeof(pe);
    pe.config   = (PERF_COUNT_HW_CACHE_LL)
                | (PERF_COUNT_HW_CACHE_OP_READ    << 8)
                | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
    pe.disabled       = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv     = 1;
    return (int)syscall(SYS_perf_event_open, &pe, 0, -1, -1, 0);
}

static void pstart(int fd) { ioctl(fd, PERF_EVENT_IOC_RESET, 0); ioctl(fd, PERF_EVENT_IOC_ENABLE, 0); }
static void pstop (int fd) { ioctl(fd, PERF_EVENT_IOC_DISABLE, 0); }
static uint64_t pread_fd(int fd) { uint64_t v = 0; read(fd, &v, 8); return v; }

static inline int64_t tv_diff_us(const struct timeval& a, const struct timeval& b)
{
    return (b.tv_sec * 1000000LL + b.tv_usec) - (a.tv_sec * 1000000LL + a.tv_usec);
}

// ── Per-thread argument + result struct ──────────────────────────────────────

struct PerfWorkerArgs {
    // ── scan inputs ──
    const IVFIndex* idx;
    const float*    base;
    const float*    query;
    size_t          d;
    size_t          k;

    // IVF_FLATTEN == 0: cluster-split
    const uint32_t* probe_ids;
    size_t          nprobe;
    int             tid;
    int             nth;

    // IVF_FLATTEN == 1: flatten-split
    const uint32_t* flat_ids;
    const size_t*   flat_pos;
    size_t          f_start;
    size_t          f_end;

    // ── outputs ──
    std::priority_queue<std::pair<float, uint32_t>> local_heap;
    size_t   vec_scanned;   // vectors actually processed this call
    uint64_t cyc;
    uint64_t ins;
    uint64_t llc_miss;
    int      perf_ok;
};

// ── Worker function ───────────────────────────────────────────────────────────

static void* perf_worker(void* arg)
{
    auto* a = static_cast<PerfWorkerArgs*>(arg);
    auto& heap = a->local_heap;
    const float*    q   = a->query;
    const size_t    d   = a->d;
    const size_t    k   = a->k;
    const IVFIndex& idx = *a->idx;

    // Open FDs from inside this thread (pid=0 = this thread)
    int fd_cyc = perf_open_thread(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
    int fd_ins = perf_open_thread(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
    int fd_llc = perf_llc_miss_thread();
    a->perf_ok = (fd_cyc >= 0 && fd_ins >= 0 && fd_llc >= 0);

    if (a->perf_ok) { pstart(fd_cyc); pstart(fd_ins); pstart(fd_llc); }

    size_t count = 0;

#if IVF_FLATTEN
    // ── Flatten mode: scan slice [f_start, f_end) of the flat vector list ────
    if (a->flat_pos != nullptr) {
        for (size_t j = a->f_start; j < a->f_end; ++j) {
            size_t   pos  = a->flat_pos[j];
            uint32_t orig = a->flat_ids[j];
            float ip  = simd_inner_product_neon_unroll(
                idx.reordered_base.data() + pos * d, q, d);
            float dis = 1.0f - ip;
            if (heap.size() < k)                 { heap.push({dis, orig}); }
            else if (dis < heap.top().first)     { heap.push({dis, orig}); heap.pop(); }
        }
        count = a->f_end - a->f_start;
    } else {
        for (size_t j = a->f_start; j < a->f_end; ++j) {
            uint32_t orig = a->flat_ids[j];
            float ip  = simd_inner_product_neon_unroll(
                a->base + (size_t)orig * d, q, d);
            float dis = 1.0f - ip;
            if (heap.size() < k)                 { heap.push({dis, orig}); }
            else if (dis < heap.top().first)     { heap.push({dis, orig}); heap.pop(); }
        }
        count = a->f_end - a->f_start;
    }
#else
    // ── Cluster-split mode: round-robin cluster assignment ───────────────────
    for (size_t ci = (size_t)a->tid; ci < a->nprobe; ci += (size_t)a->nth) {
        uint32_t c = a->probe_ids[ci];
        if (idx.reordered) {
            size_t start = idx.cluster_offset[c];
            size_t end   = idx.cluster_offset[c + 1];
            for (size_t j = start; j < end; ++j) {
                float ip  = simd_inner_product_neon_unroll(
                    idx.reordered_base.data() + j * d, q, d);
                float dis = 1.0f - ip;
                uint32_t orig = idx.invlists[c][j - start];
                if (heap.size() < k)             { heap.push({dis, orig}); }
                else if (dis < heap.top().first) { heap.push({dis, orig}); heap.pop(); }
                ++count;
            }
        } else {
            for (uint32_t orig : idx.invlists[c]) {
                float ip  = simd_inner_product_neon_unroll(
                    a->base + (size_t)orig * d, q, d);
                float dis = 1.0f - ip;
                if (heap.size() < k)             { heap.push({dis, orig}); }
                else if (dis < heap.top().first) { heap.push({dis, orig}); heap.pop(); }
                ++count;
            }
        }
    }
#endif

    if (a->perf_ok) { pstop(fd_cyc); pstop(fd_ins); pstop(fd_llc); }

    a->vec_scanned = count;
    if (a->perf_ok) {
        a->cyc      = pread_fd(fd_cyc);
        a->ins      = pread_fd(fd_ins);
        a->llc_miss = pread_fd(fd_llc);
        close(fd_cyc); close(fd_ins); close(fd_llc);
    }
    return nullptr;
}

// ── Data loader ───────────────────────────────────────────────────────────────

template<typename T>
static T* load_data(const std::string& path, size_t& n, size_t& d)
{
    std::ifstream fin(path, std::ios::binary);
    fin.read((char*)&n, 4);
    fin.read((char*)&d, 4);
    T* data = new T[n * d];
    for (size_t i = 0; i < n; ++i)
        fin.read((char*)data + i * d * sizeof(T), d * sizeof(T));
    std::cerr << "[load] " << path << "  n=" << n << "  d=" << d << "\n";
    return data;
}

// =============================================================================
int main(int argc, char** argv)
{
    const size_t nprobe     = (argc > 1) ? (size_t)atoi(argv[1]) : IVF_NPROBE;
    const int    num_threads = NUM_THREADS;
    const size_t k           = 10;

    std::cerr << "IVF_FLATTEN=" << IVF_FLATTEN
              << "  nlist="  << IVF_NLIST
              << "  nprobe=" << nprobe
              << "  reorder=" << IVF_REORDER
              << "  threads=" << num_threads << "\n";

    // ── Load dataset ─────────────────────────────────────────────────────────
    const std::string dp = "/anndata/";
    size_t test_number = 0, base_number = 0, gt_d = 0, vecdim = 0;
    float* queries = load_data<float>(dp + "DEEP100K.query.fbin",       test_number, vecdim);
    int*   gt      = load_data<int>  (dp + "DEEP100K.gt.query.100k.top100.bin", test_number, gt_d);
    float* base    = load_data<float>(dp + "DEEP100K.base.100k.fbin",   base_number, vecdim);

    // ── Build / load IVF index ───────────────────────────────────────────────
    IVFIndex idx;
    char cache_path[256];
    snprintf(cache_path, sizeof(cache_path),
             "files/ivf_nlist%d_reorder%d.bin", IVF_NLIST, IVF_REORDER);

    struct timeval tb0, tb1;
    gettimeofday(&tb0, NULL);
    if (ivf_load(idx, cache_path)) {
        gettimeofday(&tb1, NULL);
        std::cerr << "[build] loaded from cache " << cache_path
                  << "  (" << tv_diff_us(tb0, tb1) / 1000 << " ms)\n";
    } else {
        ivf_build(idx, base, base_number, vecdim, IVF_NLIST, 25, IVF_REORDER != 0);
        gettimeofday(&tb1, NULL);
        std::cerr << "[build] built IVFIndex  ("
                  << tv_diff_us(tb0, tb1) / 1000 << " ms)\n";
        ivf_save(idx, cache_path);
    }

    // ── Accumulators (per-thread, summed over PROFILE_QUERIES queries) ────────
    std::vector<uint64_t> sum_cyc     (num_threads, 0);
    std::vector<uint64_t> sum_ins     (num_threads, 0);
    std::vector<uint64_t> sum_llc     (num_threads, 0);
    std::vector<uint64_t> sum_vecs    (num_threads, 0);
    bool perf_supported = true;

    int64_t t_coarse_us = 0, t_scan_us = 0, t_merge_us = 0;
    float   total_recall = 0.0f;
    int     measured     = 0;

    const int total_queries = WARMUP_QUERIES + PROFILE_QUERIES;

    // ── Query loop ────────────────────────────────────────────────────────────
    for (int qi = 0; qi < total_queries && qi < (int)test_number; ++qi)
    {
        const float* query = queries + qi * vecdim;
        const bool   measure = (qi >= WARMUP_QUERIES);

        struct timeval tp0, tp1;

        // ── Phase 1: Coarse ──────────────────────────────────────────────────
        if (measure) gettimeofday(&tp0, NULL);

        std::vector<std::pair<float, uint32_t>> coarse(idx.nlist);
        for (size_t c = 0; c < idx.nlist; ++c) {
            float ip = simd_inner_product_neon_unroll(
                idx.centroids.data() + c * vecdim, query, vecdim);
            coarse[c] = {1.0f - ip, static_cast<uint32_t>(c)};
        }
        const size_t np = std::min(nprobe, idx.nlist);
        std::partial_sort(coarse.begin(), coarse.begin() + np, coarse.end());
        std::vector<uint32_t> probe_ids(np);
        for (size_t i = 0; i < np; ++i) probe_ids[i] = coarse[i].second;

        if (measure) { gettimeofday(&tp1, NULL); t_coarse_us += tv_diff_us(tp0, tp1); }

        // ── Phase 2: Scan (parallel) ─────────────────────────────────────────
        if (measure) gettimeofday(&tp0, NULL);

        std::vector<pthread_t>       tids(num_threads);
        std::vector<PerfWorkerArgs>  args(num_threads);

#if IVF_FLATTEN
        size_t total_vecs = 0;
        for (size_t i = 0; i < np; ++i)
            total_vecs += idx.invlists[probe_ids[i]].size();

        std::vector<uint32_t> flat_ids;
        std::vector<size_t>   flat_pos;
        flat_ids.reserve(total_vecs);
        if (idx.reordered) flat_pos.reserve(total_vecs);

        for (size_t i = 0; i < np; ++i) {
            uint32_t c = probe_ids[i];
            size_t base_pos = idx.reordered ? idx.cluster_offset[c] : 0;
            for (size_t j = 0; j < idx.invlists[c].size(); ++j) {
                flat_ids.push_back(idx.invlists[c][j]);
                if (idx.reordered) flat_pos.push_back(base_pos + j);
            }
        }

        size_t chunk = (total_vecs + (size_t)num_threads - 1) / (size_t)num_threads;
        for (int t = 0; t < num_threads; ++t) {
            args[t].idx      = &idx;
            args[t].base     = base;
            args[t].query    = query;
            args[t].d        = vecdim;
            args[t].k        = k;
            args[t].flat_ids = flat_ids.data();
            args[t].flat_pos = idx.reordered ? flat_pos.data() : nullptr;
            args[t].f_start  = std::min((size_t)t * chunk, total_vecs);
            args[t].f_end    = std::min((size_t)(t + 1) * chunk, total_vecs);
            pthread_create(&tids[t], nullptr, perf_worker, &args[t]);
        }
#else
        for (int t = 0; t < num_threads; ++t) {
            args[t].idx       = &idx;
            args[t].base      = base;
            args[t].query     = query;
            args[t].d         = vecdim;
            args[t].k         = k;
            args[t].probe_ids = probe_ids.data();
            args[t].nprobe    = np;
            args[t].tid       = t;
            args[t].nth       = num_threads;
            pthread_create(&tids[t], nullptr, perf_worker, &args[t]);
        }
#endif

        for (int t = 0; t < num_threads; ++t) pthread_join(tids[t], nullptr);

        if (measure) { gettimeofday(&tp1, NULL); t_scan_us += tv_diff_us(tp0, tp1); }

        // ── Phase 3: Merge ───────────────────────────────────────────────────
        if (measure) gettimeofday(&tp0, NULL);

        std::priority_queue<std::pair<float, uint32_t>> result;
        for (int t = 0; t < num_threads; ++t) {
            auto& lh = args[t].local_heap;
            while (!lh.empty()) {
                auto top = lh.top(); lh.pop();
                if (result.size() < k)                  { result.push(top); }
                else if (top.first < result.top().first){ result.push(top); result.pop(); }
            }
        }

        if (measure) { gettimeofday(&tp1, NULL); t_merge_us += tv_diff_us(tp0, tp1); }

        // ── Accumulate per-thread perf counters ──────────────────────────────
        if (measure) {
            for (int t = 0; t < num_threads; ++t) {
                sum_vecs[t] += args[t].vec_scanned;
                if (args[t].perf_ok) {
                    sum_cyc[t] += args[t].cyc;
                    sum_ins[t] += args[t].ins;
                    sum_llc[t] += args[t].llc_miss;
                } else {
                    perf_supported = false;
                }
            }
        }

        // ── Recall ───────────────────────────────────────────────────────────
        if (measure) {
            std::set<uint32_t> gtset;
            for (int j = 0; j < (int)k; ++j) gtset.insert((uint32_t)gt[j + qi * gt_d]);
            size_t acc = 0;
            while (!result.empty()) {
                if (gtset.count(result.top().second)) ++acc;
                result.pop();
            }
            total_recall += (float)acc / k;
            ++measured;
        }
    }

    // ── Report ────────────────────────────────────────────────────────────────
    const double N = (double)measured;

    std::cout << "\n";
    std::cout << "════════════════════════════════════════════════════════════\n";
    std::cout << " IVF_SIMD_CLUSTER_PTHREAD  perf profile\n";
    std::cout << " IVF_FLATTEN=" << IVF_FLATTEN
              << "  nprobe="  << nprobe
              << "  threads=" << num_threads
              << "  queries=" << measured << "\n";
    std::cout << "════════════════════════════════════════════════════════════\n";

    std::cout << std::fixed << std::setprecision(4)
              << "recall@10 :   " << total_recall / N << "\n\n";

    // Phase timings
    std::cout << std::setprecision(2)
              << "phase timings (avg per query):\n"
              << "  coarse  : " << t_coarse_us / N << " us\n"
              << "  scan    : " << t_scan_us   / N << " us  (create + work + join)\n"
              << "  merge   : " << t_merge_us  / N << " us\n"
              << "  total   : " << (t_coarse_us + t_scan_us + t_merge_us) / N << " us\n\n";

    // Per-thread breakdown
    std::cout << "per-thread breakdown (avg per query):\n";
    if (perf_supported) {
        std::cout << std::setw(6)  << "thread"
                  << std::setw(10) << "vecs"
                  << std::setw(12) << "cycles"
                  << std::setw(14) << "instructions"
                  << std::setw(12) << "LLC-misses"
                  << std::setw(8)  << "MPKI"
                  << std::setw(10) << "cyc/vec"
                  << "\n";
        std::cout << std::string(72, '-') << "\n";

        uint64_t tot_vecs = 0, tot_cyc = 0, tot_ins = 0, tot_llc = 0;
        for (int t = 0; t < num_threads; ++t) {
            double vecs    = sum_vecs[t] / N;
            double cyc     = sum_cyc [t] / N;
            double ins_val = sum_ins [t] / N;
            double llc     = sum_llc [t] / N;
            double mpki    = (ins_val > 0) ? (llc / (ins_val / 1000.0)) : 0.0;
            double cpv     = (vecs   > 0) ? (cyc / vecs) : 0.0;

            std::cout << std::setw(6)  << t
                      << std::setw(10) << std::setprecision(1) << vecs
                      << std::setw(12) << std::setprecision(1) << cyc
                      << std::setw(14) << std::setprecision(1) << ins_val
                      << std::setw(12) << std::setprecision(1) << llc
                      << std::setw(8)  << std::setprecision(2) << mpki
                      << std::setw(10) << std::setprecision(2) << cpv
                      << "\n";

            tot_vecs += sum_vecs[t];
            tot_cyc  += sum_cyc [t];
            tot_ins  += sum_ins [t];
            tot_llc  += sum_llc [t];
        }

        std::cout << std::string(72, '-') << "\n";
        double tv = tot_vecs / N, tc = tot_cyc / N;
        double ti = tot_ins  / N, tl = tot_llc / N;
        double mpki_tot = (ti > 0) ? (tl / (ti / 1000.0)) : 0.0;
        double cpv_avg  = (tv > 0) ? (tc / tv) : 0.0;
        std::cout << std::setw(6)  << "total"
                  << std::setw(10) << std::setprecision(1) << tv
                  << std::setw(12) << std::setprecision(1) << tc
                  << std::setw(14) << std::setprecision(1) << ti
                  << std::setw(12) << std::setprecision(1) << tl
                  << std::setw(8)  << std::setprecision(2) << mpki_tot
                  << std::setw(10) << std::setprecision(2) << cpv_avg
                  << "\n\n";

        // Load imbalance metric
        double mean_vecs = tv / num_threads;
        double max_vecs  = 0;
        for (int t = 0; t < num_threads; ++t)
            max_vecs = std::max(max_vecs, (double)sum_vecs[t] / N);
        std::cout << std::setprecision(1)
                  << "load balance:  mean=" << mean_vecs
                  << "  max=" << max_vecs
                  << "  imbalance=" << std::setprecision(2)
                  << ((max_vecs / mean_vecs - 1.0) * 100.0) << "%\n";
    } else {
        // perf blocked: show vec counts only
        std::cerr << "[perf] perf_event_open blocked — "
                     "check /proc/sys/kernel/perf_event_paranoid (need <= 2)\n"
                     "[perf] showing vector counts only\n\n";
        std::cout << std::setw(8) << "thread" << std::setw(12) << "vecs/query\n";
        std::cout << std::string(20, '-') << "\n";
        double max_vecs = 0, mean_vecs = 0;
        for (int t = 0; t < num_threads; ++t) {
            double v = sum_vecs[t] / N;
            std::cout << std::setw(8) << t << std::setw(12) << std::setprecision(1) << v << "\n";
            max_vecs   = std::max(max_vecs, v);
            mean_vecs += v;
        }
        mean_vecs /= num_threads;
        std::cout << "\nload balance:  mean=" << std::setprecision(1) << mean_vecs
                  << "  max=" << max_vecs
                  << "  imbalance=" << std::setprecision(2)
                  << ((max_vecs / mean_vecs - 1.0) * 100.0) << "%\n";
    }

    std::cout << "════════════════════════════════════════════════════════════\n";

    delete[] queries;
    delete[] gt;
    delete[] base;
    return 0;
}
