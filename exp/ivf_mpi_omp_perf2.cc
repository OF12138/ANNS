// =============================================================================
// exp/ivf_mpi_omp_perf2.cc — Hardware-counter profiling for Plan II hybrid
//                             (query-batch-MPI × OMP fine scan)
//
// Profiling dimensions:
//   1. Phase breakdown (per process)  — coarse / fine(OMP) wall times per query
//   2. MPI communication cost         — one MPI_Gather at end, amortized per query
//   3. Per-OMP-thread HW counters     — cycles, instructions, LLC misses,
//                                       MPKI, cycles/vec  (rank 0, representative)
//   4. Intra-process OMP load balance — vecs per OMP thread (rank 0)
//   5. Cross-process load balance     — vecs and fine-time per MPI process
//                                       (MPI_Reduce MAX/MIN/SUM to rank 0)
//
// Key difference vs Plan I profiling (ivf_mpi_omp_perf.cc):
//   Plan I:  P × T workers collaborate on each query → per-query MPI (Bcast+Gather)
//   Plan II: each process handles its own query slice independently → one final Gather
//            Communication cost is O(1) per batch, amortized over Q/P queries/process
//
// Compile (from project root):
//   mpicxx exp/ivf_mpi_omp_perf2.cc -o exp/ivf_mpi_omp_perf2 \
//          -O2 -std=c++11 -fopenmp -I. -lm
//
// Run via PBS:
//   bash exp/run_mpi_omp_perf2.sh
// =============================================================================

#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <unistd.h>

#include <mpi.h>
#include <omp.h>

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
#include <cstdint>
#include <cstdio>
#include <cfloat>

#include "ARM/Alg_parallel/ivf_flat_simd.h"

// ── Config ────────────────────────────────────────────────────────────────────
#define IVF_NLIST        1024
#define IVF_NPROBE       16
#define IVF_REORDER      0
// Queries per process: first WARMUP_QUERIES are discarded, next PROFILE_QUERIES measured.
// Total queries needed: P × (WARMUP + PROFILE). For P=8 → 8×250=2000 (fits DEEP100K).
#define WARMUP_QUERIES   50
#define PROFILE_QUERIES  200

// ── perf helpers (same as ivf_mpi_omp_perf.cc) ───────────────────────────────
static int perf_open_hw(uint32_t config)
{
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.type           = PERF_TYPE_HARDWARE;
    pe.size           = sizeof(pe);
    pe.config         = config;
    pe.disabled       = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv     = 1;
    return (int)syscall(SYS_perf_event_open, &pe, 0, -1, -1, 0);
}

static int perf_open_llc()
{
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.type   = PERF_TYPE_HW_CACHE;
    pe.size   = sizeof(pe);
    pe.config = (PERF_COUNT_HW_CACHE_LL)
              | (PERF_COUNT_HW_CACHE_OP_READ     << 8)
              | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
    pe.disabled       = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv     = 1;
    return (int)syscall(SYS_perf_event_open, &pe, 0, -1, -1, 0);
}

static void pstart(int fd) { ioctl(fd, PERF_EVENT_IOC_RESET,0); ioctl(fd, PERF_EVENT_IOC_ENABLE,0); }
static void pstop (int fd) { ioctl(fd, PERF_EVENT_IOC_DISABLE,0); }
static uint64_t pread_fd(int fd) { uint64_t v=0; read(fd,&v,8); return v; }

template<typename T>
static T* load_data(const std::string& path, size_t& n, size_t& d)
{
    std::ifstream fin(path, std::ios::binary);
    fin.read((char*)&n, 4);
    fin.read((char*)&d, 4);
    T* data = new T[n * d];
    for (size_t i = 0; i < n; ++i)
        fin.read((char*)data + i*d*sizeof(T), d*sizeof(T));
    return data;
}

static inline int64_t tv_diff_us(const struct timeval& a, const struct timeval& b)
{
    return (b.tv_sec*1000000LL + b.tv_usec) - (a.tv_sec*1000000LL + a.tv_usec);
}


// =============================================================================
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int    nthreads = omp_get_max_threads();
    const size_t nprobe   = (argc > 1) ? (size_t)atoi(argv[1]) : IVF_NPROBE;
    const size_t k        = 10;

    // Each process handles TOTAL_PER_PROC consecutive queries from the query file.
    // Rank r uses queries [ r*TOTAL_PER_PROC .. (r+1)*TOTAL_PER_PROC ).
    const int TOTAL_PER_PROC = WARMUP_QUERIES + PROFILE_QUERIES;
    const int lo_q = rank * TOTAL_PER_PROC;       // absolute index into query file
    const int hi_q = lo_q + TOTAL_PER_PROC;

    char hostname[64] = {};
    gethostname(hostname, sizeof(hostname));

    if (rank == 0) {
        std::cerr << "════════════════════════════════════════════════════\n";
        std::cerr << " IVF_MPI_OMP_PLAN2 — perf profiling\n";
        std::cerr << " mpi_procs=" << size
                  << "  omp_threads=" << nthreads
                  << "  nprobe=" << nprobe << "\n";
        std::cerr << " queries/proc=" << TOTAL_PER_PROC
                  << "  (warmup=" << WARMUP_QUERIES
                  << "  profile=" << PROFILE_QUERIES << ")\n";
        std::cerr << "════════════════════════════════════════════════════\n";
    }
    for (int r = 0; r < size; ++r) {
        if (rank == r)
            std::cerr << "  rank " << rank << " → " << hostname
                      << "  queries [" << lo_q << ".." << hi_q << ")\n";
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // ── Load data ─────────────────────────────────────────────────────────────
    const std::string dp = "/anndata/";
    size_t total_queries = 0, base_number = 0, gt_d = 0, vecdim = 0;
    float* queries_all = load_data<float>(dp + "DEEP100K.query.fbin",       total_queries, vecdim);
    int*   gt_all      = load_data<int>  (dp + "DEEP100K.gt.query.100k.top100.bin", total_queries, gt_d);
    float* base        = load_data<float>(dp + "DEEP100K.base.100k.fbin",   base_number, vecdim);

    // ── Build / load IVF index ────────────────────────────────────────────────
    IVFIndex idx;
    char cache_path[256];
    snprintf(cache_path, sizeof(cache_path),
             "files/ivf_nlist%d_reorder%d.bin", IVF_NLIST, IVF_REORDER);

    bool loaded = ivf_load(idx, cache_path);
    if (!loaded) {
        if (rank == 0) {
            ivf_build(idx, base, base_number, vecdim, IVF_NLIST, 25, IVF_REORDER != 0);
            ivf_save(idx, cache_path);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank != 0) ivf_load(idx, cache_path);
    }
    if (rank == 0)
        std::cerr << "[index] " << (loaded ? "loaded from cache" : "built") << "\n";

    // ── Check perf availability ───────────────────────────────────────────────
    bool perf_ok = true;
    { int fd = perf_open_hw(PERF_COUNT_HW_CPU_CYCLES); if (fd < 0) perf_ok = false; else close(fd); }
    if (rank == 0 && !perf_ok)
        std::cerr << "[perf] perf_event_open blocked — "
                     "check /proc/sys/kernel/perf_event_paranoid (need <= 2)\n"
                     "[perf] showing vector counts + wall-time only\n";

    // ── Accumulators — rank 0's own OMP threads ───────────────────────────────
    std::vector<uint64_t> sum_vecs(nthreads, 0);
    std::vector<uint64_t> sum_cyc (nthreads, 0);
    std::vector<uint64_t> sum_ins (nthreads, 0);
    std::vector<uint64_t> sum_llc (nthreads, 0);

    // ── Per-process scalar accumulators ──────────────────────────────────────
    double t_coarse_s = 0.0, t_fine_s = 0.0;
    double proc_vecs  = 0.0;
    float  total_recall = 0.0f;

    std::vector<float>   local_recalls  (PROFILE_QUERIES, 0.0f);
    std::vector<int64_t> local_latencies(PROFILE_QUERIES, 0LL);

    std::vector<uint64_t> q_vecs(nthreads, 0);
    std::vector<uint64_t> q_cyc (nthreads, 0);
    std::vector<uint64_t> q_ins (nthreads, 0);
    std::vector<uint64_t> q_llc (nthreads, 0);

    // ── Query loop (no inter-query MPI) ───────────────────────────────────────
    for (int local_i = 0; local_i < TOTAL_PER_PROC; ++local_i)
    {
        const int    qi      = lo_q + local_i;
        const float* query   = queries_all + (size_t)qi * vecdim;
        const bool   measure = (local_i >= WARMUP_QUERIES);
        const int    prof_i  = local_i - WARMUP_QUERIES;   // index into local_recalls

        struct timeval wall_t0, wall_t1;
        if (measure) gettimeofday(&wall_t0, NULL);

        // ── Coarse (single thread) ────────────────────────────────────────────
        const size_t np = std::min(nprobe, idx.nlist);
        std::vector<uint32_t> probe_ids(np);

        double t0 = omp_get_wtime();
        {
            std::vector<std::pair<float,uint32_t>> coarse(idx.nlist);
            for (size_t c = 0; c < idx.nlist; ++c) {
                float ip = simd_inner_product_neon_unroll(
                    idx.centroids.data() + c*vecdim, query, vecdim);
                coarse[c] = {1.0f - ip, (uint32_t)c};
            }
            std::partial_sort(coarse.begin(), coarse.begin()+np, coarse.end());
            for (size_t i = 0; i < np; ++i) probe_ids[i] = coarse[i].second;
        }
        if (measure) t_coarse_s += omp_get_wtime() - t0;

        // ── Fine scan (OMP parallel) with per-thread perf ────────────────────
        std::vector<std::priority_queue<std::pair<float,uint32_t>>> thread_heaps(nthreads);

        if (rank == 0 && measure)
            for (int t = 0; t < nthreads; ++t)
                q_vecs[t] = q_cyc[t] = q_ins[t] = q_llc[t] = 0;

        t0 = omp_get_wtime();

        #pragma omp parallel num_threads(nthreads)
        {
            int tid = omp_get_thread_num();
            auto& heap = thread_heaps[tid];

            int fd_c = -1, fd_i = -1, fd_l = -1;
            bool thread_perf = perf_ok && (rank == 0) && measure;
            if (thread_perf) {
                fd_c = perf_open_hw(PERF_COUNT_HW_CPU_CYCLES);
                fd_i = perf_open_hw(PERF_COUNT_HW_INSTRUCTIONS);
                fd_l = perf_open_llc();
                thread_perf = (fd_c >= 0 && fd_i >= 0 && fd_l >= 0);
            }
            if (thread_perf) { pstart(fd_c); pstart(fd_i); pstart(fd_l); }

            uint64_t vecs = 0;
            #pragma omp for schedule(dynamic, 1)
            for (int probe = 0; probe < (int)np; ++probe) {
                const uint32_t c = probe_ids[probe];
                for (uint32_t orig : idx.invlists[c]) {
                    float ip  = simd_inner_product_neon_unroll(
                        base + (size_t)orig*vecdim, query, vecdim);
                    float dis = 1.0f - ip;
                    if (heap.size() < k) heap.push({dis, orig});
                    else if (dis < heap.top().first) { heap.pop(); heap.push({dis, orig}); }
                    ++vecs;
                }
            }

            if (thread_perf) { pstop(fd_c); pstop(fd_i); pstop(fd_l); }

            if (measure) {
                q_vecs[tid] = vecs;
                if (thread_perf) {
                    q_cyc[tid] = pread_fd(fd_c);
                    q_ins[tid] = pread_fd(fd_i);
                    q_llc[tid] = pread_fd(fd_l);
                }
            }
            if (thread_perf) { close(fd_c); close(fd_i); close(fd_l); }
        }

        if (measure) {
            t_fine_s += omp_get_wtime() - t0;
            uint64_t this_vecs = 0;
            for (int t = 0; t < nthreads; ++t) {
                this_vecs += q_vecs[t];
                if (rank == 0) {
                    sum_vecs[t] += q_vecs[t];
                    sum_cyc [t] += q_cyc [t];
                    sum_ins [t] += q_ins [t];
                    sum_llc [t] += q_llc [t];
                }
            }
            proc_vecs += (double)this_vecs;
        }

        // ── Merge thread heaps → query result ─────────────────────────────────
        std::priority_queue<std::pair<float,uint32_t>> result;
        for (auto& h : thread_heaps) {
            while (!h.empty()) {
                auto top = h.top(); h.pop();
                if (result.size() < k) result.push(top);
                else if (top.first < result.top().first) { result.pop(); result.push(top); }
            }
        }

        if (measure) {
            gettimeofday(&wall_t1, NULL);
            local_latencies[prof_i] = tv_diff_us(wall_t0, wall_t1);

            std::set<uint32_t> gtset;
            for (int j = 0; j < (int)k; ++j)
                gtset.insert((uint32_t)gt_all[j + qi*(int)gt_d]);
            size_t acc = 0;
            while (!result.empty()) {
                if (gtset.count(result.top().second)) ++acc;
                result.pop();
            }
            local_recalls[prof_i] = (float)acc / k;
            total_recall += local_recalls[prof_i];
        }
    }   // end query loop

    // ── MPI_Gather: collect all results to rank 0 ─────────────────────────────
    // This is the ONLY MPI call — paid once per batch.
    const int total_queries_profiled = size * PROFILE_QUERIES;
    std::vector<float>   all_recalls;
    std::vector<int64_t> all_latencies;
    if (rank == 0) {
        all_recalls  .resize(total_queries_profiled);
        all_latencies.resize(total_queries_profiled);
    }

    double gather_t0 = MPI_Wtime();
    MPI_Gather(local_recalls  .data(), PROFILE_QUERIES, MPI_FLOAT,
               all_recalls    .data(), PROFILE_QUERIES, MPI_FLOAT,
               0, MPI_COMM_WORLD);
    MPI_Gather(local_latencies.data(), PROFILE_QUERIES, MPI_LONG_LONG,
               all_latencies  .data(), PROFILE_QUERIES, MPI_LONG_LONG,
               0, MPI_COMM_WORLD);
    double gather_s = MPI_Wtime() - gather_t0;

    // ── Cross-process aggregates ──────────────────────────────────────────────
    double fine_max=0, fine_min=0, fine_sum=0;
    double vecs_max=0, vecs_min=0, vecs_sum=0;
    double coarse_max=0, coarse_min=0, coarse_sum=0;
    double gather_max=0;

    MPI_Reduce(&t_fine_s,   &fine_max,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_fine_s,   &fine_min,   1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_fine_s,   &fine_sum,   1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_coarse_s, &coarse_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_coarse_s, &coarse_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_coarse_s, &coarse_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&proc_vecs,  &vecs_max,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&proc_vecs,  &vecs_min,   1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&proc_vecs,  &vecs_sum,   1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&gather_s,   &gather_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // ── Report (rank 0 only) ──────────────────────────────────────────────────
    if (rank == 0) {
        const double N  = (double)PROFILE_QUERIES;   // per process
        const double NQ = N * size;                   // total profiled queries
        const double us = 1e6;

        // Average recall and latency from gathered results
        double avg_recall = 0.0, avg_lat_us = 0.0;
        for (int i = 0; i < total_queries_profiled; ++i) {
            avg_recall  += all_recalls  [i];
            avg_lat_us  += (double)all_latencies[i];
        }
        avg_recall /= NQ;
        avg_lat_us /= NQ;

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "\n════════════════════════════════════════════════════\n";
        std::cout << " IVF_MPI_OMP_PLAN2 — perf profile\n";
        std::cout << " mpi_procs=" << size
                  << "  omp_threads=" << nthreads
                  << "  nprobe=" << nprobe << "\n";
        std::cout << "════════════════════════════════════════════════════\n";
        std::cout << "recall@10        : " << avg_recall << "\n";
        std::cout << "avg latency (us) : " << avg_lat_us << "  (per-query wall time)\n\n";

        // ── Phase timings ─────────────────────────────────────────────────────
        std::cout << std::setprecision(2);
        std::cout << "── Phase timings (avg/query, rank 0 view) ──\n";
        std::cout << "  coarse (single-thread) : " << t_coarse_s / N * us << " us\n";
        std::cout << "  fine   (OMP " << nthreads << " threads) : "
                  << t_fine_s   / N * us << " us\n";
        std::cout << "  total compute          : "
                  << (t_coarse_s + t_fine_s) / N * us << " us\n\n";

        // ── MPI communication (amortized) ─────────────────────────────────────
        // Plan II: ONE Gather per batch, not per query.
        // Amortized cost = gather_total / (P × PROFILE_QUERIES)
        std::cout << "── MPI communication ──\n";
        std::cout << "  gather (once per batch)         : "
                  << gather_max * 1000.0 << " ms  total\n";
        std::cout << "  gather amortized per query      : "
                  << gather_max / NQ * us << " us/query\n";
        std::cout << "  [vs Plan I per-query Gather     ~ 17-163 us/query "
                     "depending on P/nodes]\n\n";

        // ── Per-OMP-thread breakdown (rank 0) ─────────────────────────────────
        std::cout << "── Per-OMP-thread breakdown — rank 0 (avg/query) ──\n";
        if (perf_ok) {
            std::cout << std::setw(7)  << "thread"
                      << std::setw(9)  << "vecs"
                      << std::setw(11) << "cycles"
                      << std::setw(13) << "instrs"
                      << std::setw(11) << "LLC-miss"
                      << std::setw(8)  << "MPKI"
                      << std::setw(9)  << "cyc/vec"
                      << "\n";
            std::cout << std::string(68, '-') << "\n";

            uint64_t tot_vecs=0, tot_cyc=0, tot_ins=0, tot_llc=0;
            double max_vecs = 0;
            for (int t = 0; t < nthreads; ++t) {
                double vecs = sum_vecs[t] / N;
                double cyc  = sum_cyc [t] / N;
                double ins  = sum_ins [t] / N;
                double llc  = sum_llc [t] / N;
                double mpki = (ins > 0) ? (llc / (ins / 1000.0)) : 0.0;
                double cpv  = (vecs > 0) ? (cyc / vecs) : 0.0;
                max_vecs = std::max(max_vecs, vecs);
                std::cout << std::setw(7)  << t
                          << std::setw(9)  << std::setprecision(1) << vecs
                          << std::setw(11) << std::setprecision(1) << cyc
                          << std::setw(13) << std::setprecision(1) << ins
                          << std::setw(11) << std::setprecision(1) << llc
                          << std::setw(8)  << std::setprecision(2) << mpki
                          << std::setw(9)  << std::setprecision(2) << cpv
                          << "\n";
                tot_vecs += sum_vecs[t]; tot_cyc += sum_cyc[t];
                tot_ins  += sum_ins [t]; tot_llc += sum_llc[t];
            }
            std::cout << std::string(68, '-') << "\n";
            double tv=tot_vecs/N, tc=tot_cyc/N, ti=tot_ins/N, tl=tot_llc/N;
            std::cout << std::setw(7)  << "total"
                      << std::setw(9)  << std::setprecision(1) << tv
                      << std::setw(11) << std::setprecision(1) << tc
                      << std::setw(13) << std::setprecision(1) << ti
                      << std::setw(11) << std::setprecision(1) << tl
                      << std::setw(8)  << std::setprecision(2)
                      << (ti>0 ? tl/(ti/1000.0) : 0.0)
                      << std::setw(9)  << std::setprecision(2)
                      << (tv>0 ? tc/tv : 0.0) << "\n\n";

            double mean_vecs = tv / nthreads;
            std::cout << "── Intra-process OMP load balance (rank 0) ──\n";
            std::cout << "  mean vecs/thread : " << std::setprecision(1) << mean_vecs << "\n";
            std::cout << "  max  vecs/thread : " << max_vecs << "\n";
            std::cout << "  imbalance        : " << std::setprecision(2)
                      << ((mean_vecs > 0) ? (max_vecs/mean_vecs - 1.0)*100.0 : 0.0) << "%\n\n";
        } else {
            std::cout << std::setw(8) << "thread" << std::setw(12) << "vecs/query\n";
            std::cout << std::string(20, '-') << "\n";
            double max_v=0, mean_v=0;
            for (int t = 0; t < nthreads; ++t) {
                double v = sum_vecs[t] / N;
                std::cout << std::setw(8) << t << std::setw(12) << std::setprecision(1) << v << "\n";
                max_v = std::max(max_v, v); mean_v += v;
            }
            mean_v /= nthreads;
            std::cout << "\n── Intra-process OMP load balance (rank 0) ──\n";
            std::cout << "  mean : " << std::setprecision(1) << mean_v << "\n";
            std::cout << "  max  : " << max_v << "\n";
            std::cout << "  imbalance: " << std::setprecision(2)
                      << ((mean_v > 0) ? (max_v/mean_v - 1.0)*100.0 : 0.0) << "%\n\n";
        }

        // ── Cross-process load balance ─────────────────────────────────────────
        // In Plan II, imbalance comes from different query difficulties, not
        // nprobe partition — expect lower imbalance than Plan I.
        std::cout << "── Cross-process MPI load balance ──\n";
        std::cout << "  [Note: Plan II queries are pre-assigned, not work-stolen.\n"
                     "   Imbalance here reflects query difficulty variance, not\n"
                     "   structural nprobe-partition skew as in Plan I.]\n";
        std::cout << "  vecs/proc    avg=" << std::setprecision(1) << vecs_sum/size/N
                  << "  max=" << vecs_max/N
                  << "  min=" << vecs_min/N
                  << "  imbalance=" << std::setprecision(2)
                  << ((vecs_max>0) ? (vecs_max-vecs_min)/vecs_max*100.0 : 0.0) << "%\n";
        std::cout << "  coarse/proc  avg=" << std::setprecision(2) << coarse_sum/size/N*us
                  << "us  max=" << coarse_max/N*us
                  << "us  min=" << coarse_min/N*us << "us\n";
        std::cout << "  fine/proc    avg=" << fine_sum/size/N*us
                  << "us  max=" << fine_max/N*us
                  << "us  min=" << fine_min/N*us
                  << "us  imbalance=" << std::setprecision(2)
                  << ((fine_max>0) ? (fine_max-fine_min)/fine_max*100.0 : 0.0) << "%\n";
        std::cout << "════════════════════════════════════════════════════\n";
    }

    delete[] queries_all;
    delete[] gt_all;
    delete[] base;
    MPI_Finalize();
    return 0;
}
