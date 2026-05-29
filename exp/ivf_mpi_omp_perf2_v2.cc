// =============================================================================
// exp/ivf_mpi_omp_perf2_v2.cc — Hardware-counter profiling for Plan II v2
//                                (query-batch-MPI × OMP fine scan, REORDERED)
//
// Identical to ivf_mpi_omp_perf2.cc except:
//   IVF_REORDER = 1  — index loaded from reorder=1 cache
//   Fine scan reads idx.reordered_base[j*d]  (contiguous, sequential)
//     instead of base[orig*d]               (random scatter across 38 MB)
//
// Purpose: measure the cache improvement from data reordering.
// Compare output directly against ivf_mpi_omp_perf2 to see:
//   LLC miss rate    77%  →  expected 15–30%
//   IPC              0.37 →  expected 1.0–2.0
//   cyc/vec          487  →  expected 80–150
//   fine scan time   ~500µs avg → expected 80–200µs avg
//
// Compile (from project root):
//   mpicxx exp/ivf_mpi_omp_perf2_v2.cc -o exp/ivf_mpi_omp_perf2_v2 \
//          -O2 -std=c++11 -fopenmp -I. -lm
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
#define IVF_REORDER      1          // ← reordering enabled
#define WARMUP_QUERIES   50
#define PROFILE_QUERIES  200


// =============================================================================
// perf helpers  (identical to ivf_mpi_omp_perf2.cc)
// =============================================================================

static int perf_hw(uint64_t config)
{
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.type = PERF_TYPE_HARDWARE; pe.size = sizeof(pe); pe.config = config;
    pe.disabled = 1; pe.exclude_kernel = 1; pe.exclude_hv = 1;
    return (int)syscall(SYS_perf_event_open, &pe, 0, -1, -1, 0);
}

static int perf_cache(uint32_t cache_id, uint32_t op, uint32_t result)
{
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.type   = PERF_TYPE_HW_CACHE;
    pe.size   = sizeof(pe);
    pe.config = cache_id | (op << 8) | (result << 16);
    pe.disabled = 1; pe.exclude_kernel = 1; pe.exclude_hv = 1;
    return (int)syscall(SYS_perf_event_open, &pe, 0, -1, -1, 0);
}

static void pstart(int fd) { ioctl(fd, PERF_EVENT_IOC_RESET,0); ioctl(fd, PERF_EVENT_IOC_ENABLE,0); }
static void pstop (int fd) { ioctl(fd, PERF_EVENT_IOC_DISABLE,0); }
static uint64_t pread_fd(int fd) { uint64_t v=0; if(fd>=0) read(fd,&v,8); return v; }
static void pclose(int fd) { if(fd>=0) close(fd); }


struct ThreadCounters {
    uint64_t vecs=0, cyc=0, ins=0;
    uint64_t l1m=0, l1a=0, llcm=0, llca=0, brm=0, bri=0;
    bool ok=false;
};

template<typename T>
static T* load_data(const std::string& path, size_t& n, size_t& d)
{
    std::ifstream fin(path, std::ios::binary);
    fin.read((char*)&n, 4); fin.read((char*)&d, 4);
    T* data = new T[n * d];
    for (size_t i = 0; i < n; ++i)
        fin.read((char*)data + i*d*sizeof(T), d*sizeof(T));
    return data;
}

static inline int64_t tv_diff_us(const struct timeval& a, const struct timeval& b)
{ return (b.tv_sec*1000000LL+b.tv_usec)-(a.tv_sec*1000000LL+a.tv_usec); }

static void print_sep(int w) { std::cout << std::string(w, '-') << "\n"; }
static double rate(uint64_t miss, uint64_t total)
{ return (total > 0) ? 100.0 * miss / total : -1.0; }


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

    const int TOTAL_PER_PROC = WARMUP_QUERIES + PROFILE_QUERIES;
    const int lo_q = rank * TOTAL_PER_PROC;
    const int hi_q = lo_q + TOTAL_PER_PROC;

    char hostname[64] = {};
    gethostname(hostname, sizeof(hostname));

    if (rank == 0) {
        std::cerr << "════════════════════════════════════════════════════\n";
        std::cerr << " IVF_MPI_OMP_PLAN2_V2 (reordered) — perf profiling\n";
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
    size_t total_queries=0, base_number=0, gt_d=0, vecdim=0;
    float* queries_all = load_data<float>(dp+"DEEP100K.query.fbin",                   total_queries, vecdim);
    int*   gt_all      = load_data<int>  (dp+"DEEP100K.gt.query.100k.top100.bin",     total_queries, gt_d);
    float* base        = load_data<float>(dp+"DEEP100K.base.100k.fbin",               base_number, vecdim);

    // ── Build / load reordered index ──────────────────────────────────────────
    IVFIndex idx;
    char cache_path[256];
    snprintf(cache_path, sizeof(cache_path),
             "files/ivf_nlist%d_reorder%d.bin", IVF_NLIST, IVF_REORDER);
    bool loaded = ivf_load(idx, cache_path);
    if (!loaded) {
        if (rank == 0) {
            ivf_build(idx, base, base_number, vecdim, IVF_NLIST, 25, true);  // reorder=true
            ivf_save(idx, cache_path);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank != 0) ivf_load(idx, cache_path);
    }
    if (rank == 0) std::cerr << "[index] reorder=" << idx.reordered
                             << "  " << (loaded ? "loaded" : "built") << "\n";

    // ── Probe counter availability ────────────────────────────────────────────
    bool have_cyc = (perf_hw(PERF_COUNT_HW_CPU_CYCLES) >= 0);
    bool have_l1  = (perf_cache(PERF_COUNT_HW_CACHE_L1D,
                        PERF_COUNT_HW_CACHE_OP_READ,
                        PERF_COUNT_HW_CACHE_RESULT_MISS) >= 0);
    bool have_llc = (perf_cache(PERF_COUNT_HW_CACHE_LL,
                        PERF_COUNT_HW_CACHE_OP_READ,
                        PERF_COUNT_HW_CACHE_RESULT_MISS) >= 0);
    bool have_br  = (perf_hw(PERF_COUNT_HW_BRANCH_MISSES) >= 0);

    if (rank == 0) {
        std::cerr << "[perf] cycles=" << have_cyc << "  L1=" << have_l1
                  << "  LLC=" << have_llc << "  branch=" << have_br << "\n";
        if (!have_cyc)
            std::cerr << "[perf] WARNING: hardware counters blocked "
                         "(need /proc/sys/kernel/perf_event_paranoid <= 2)\n";
    }

    // ── Accumulators ──────────────────────────────────────────────────────────
    std::vector<ThreadCounters> thread_acc(nthreads);
    double t_coarse_s=0, t_fine_s=0, proc_vecs=0;
    float  total_recall=0;

    std::vector<float>   local_recalls  (PROFILE_QUERIES, 0.0f);
    std::vector<int64_t> local_latencies(PROFILE_QUERIES, 0LL);
    std::vector<ThreadCounters> q_tc(nthreads);

    // ── Query loop ────────────────────────────────────────────────────────────
    for (int local_i = 0; local_i < TOTAL_PER_PROC; ++local_i)
    {
        const int    qi      = lo_q + local_i;
        const float* query   = queries_all + (size_t)qi * vecdim;
        const bool   measure = (local_i >= WARMUP_QUERIES);
        const int    prof_i  = local_i - WARMUP_QUERIES;

        struct timeval wall_t0, wall_t1;
        if (measure) gettimeofday(&wall_t0, NULL);

        // ── Coarse (unchanged from v1) ────────────────────────────────────────
        const size_t np = std::min(nprobe, idx.nlist);
        std::vector<uint32_t> probe_ids(np);
        double t0 = omp_get_wtime();
        {
            std::vector<std::pair<float,uint32_t>> coarse(idx.nlist);
            for (size_t c = 0; c < idx.nlist; ++c) {
                float ip = simd_inner_product_neon_unroll(
                    idx.centroids.data()+c*vecdim, query, vecdim);
                coarse[c] = {1.0f-ip, (uint32_t)c};
            }
            std::partial_sort(coarse.begin(), coarse.begin()+np, coarse.end());
            for (size_t i = 0; i < np; ++i) probe_ids[i] = coarse[i].second;
        }
        if (measure) t_coarse_s += omp_get_wtime() - t0;

        // ── Fine scan — REORDERED path + perf ────────────────────────────────
        std::vector<std::priority_queue<std::pair<float,uint32_t>>> thread_heaps(nthreads);
        if (rank == 0 && measure)
            for (int t = 0; t < nthreads; ++t) q_tc[t] = ThreadCounters{};

        const float* rb = idx.reordered_base.data();   // cluster-contiguous base

        t0 = omp_get_wtime();
        #pragma omp parallel num_threads(nthreads)
        {
            int tid = omp_get_thread_num();
            auto& heap = thread_heaps[tid];

            const bool do_perf = have_cyc && (rank == 0) && measure;
            int fd_cyc=-1, fd_ins=-1, fd_l1m=-1, fd_l1a=-1;
            int fd_llcm=-1, fd_llca=-1, fd_brm=-1, fd_bri=-1;

            if (do_perf) {
                fd_cyc  = perf_hw(PERF_COUNT_HW_CPU_CYCLES);
                fd_ins  = perf_hw(PERF_COUNT_HW_INSTRUCTIONS);
                fd_l1m  = have_l1  ? perf_cache(PERF_COUNT_HW_CACHE_L1D,
                              PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS)  : -1;
                fd_l1a  = have_l1  ? perf_cache(PERF_COUNT_HW_CACHE_L1D,
                              PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_ACCESS): -1;
                fd_llcm = have_llc ? perf_cache(PERF_COUNT_HW_CACHE_LL,
                              PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS)  : -1;
                fd_llca = have_llc ? perf_cache(PERF_COUNT_HW_CACHE_LL,
                              PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_ACCESS): -1;
                fd_brm  = have_br  ? perf_hw(PERF_COUNT_HW_BRANCH_MISSES)                    : -1;
                fd_bri  = have_br  ? perf_hw(PERF_COUNT_HW_BRANCH_INSTRUCTIONS)              : -1;

                if (fd_cyc>=0)  pstart(fd_cyc);
                if (fd_ins>=0)  pstart(fd_ins);
                if (fd_l1m>=0)  pstart(fd_l1m);
                if (fd_l1a>=0)  pstart(fd_l1a);
                if (fd_llcm>=0) pstart(fd_llcm);
                if (fd_llca>=0) pstart(fd_llca);
                if (fd_brm>=0)  pstart(fd_brm);
                if (fd_bri>=0)  pstart(fd_bri);
            }

            uint64_t vecs = 0;
            #pragma omp for schedule(dynamic, 1)
            for (int probe = 0; probe < (int)np; ++probe) {
                const uint32_t c     = probe_ids[probe];
                const size_t   start = idx.cluster_offset[c];
                const size_t   end   = idx.cluster_offset[c + 1];
                // ── KEY CHANGE vs perf2.cc ────────────────────────────────────
                // Sequential read: rb + j*vecdim advances by one vector stride
                // each iteration.  HW prefetcher issues next cache lines ahead.
                for (size_t j = start; j < end; ++j) {
                    float    ip   = simd_inner_product_neon_unroll(
                                        rb + j*vecdim, query, vecdim);
                    float    dis  = 1.0f - ip;
                    uint32_t orig = idx.invlists[c][j - start];
                    if (heap.size()<k) heap.push({dis,orig});
                    else if (dis<heap.top().first){heap.pop();heap.push({dis,orig});}
                    ++vecs;
                }
            }

            if (do_perf) {
                if (fd_cyc>=0)  pstop(fd_cyc);
                if (fd_ins>=0)  pstop(fd_ins);
                if (fd_l1m>=0)  pstop(fd_l1m);
                if (fd_l1a>=0)  pstop(fd_l1a);
                if (fd_llcm>=0) pstop(fd_llcm);
                if (fd_llca>=0) pstop(fd_llca);
                if (fd_brm>=0)  pstop(fd_brm);
                if (fd_bri>=0)  pstop(fd_bri);
            }

            if (rank == 0 && measure) {
                q_tc[tid].ok   = do_perf;
                q_tc[tid].vecs = vecs;
                q_tc[tid].cyc  = pread_fd(fd_cyc);
                q_tc[tid].ins  = pread_fd(fd_ins);
                q_tc[tid].l1m  = pread_fd(fd_l1m);
                q_tc[tid].l1a  = pread_fd(fd_l1a);
                q_tc[tid].llcm = pread_fd(fd_llcm);
                q_tc[tid].llca = pread_fd(fd_llca);
                q_tc[tid].brm  = pread_fd(fd_brm);
                q_tc[tid].bri  = pread_fd(fd_bri);
            } else if (measure) {
                q_tc[tid].vecs = vecs;
            }

            pclose(fd_cyc); pclose(fd_ins);
            pclose(fd_l1m); pclose(fd_l1a);
            pclose(fd_llcm);pclose(fd_llca);
            pclose(fd_brm); pclose(fd_bri);
        }

        if (measure) {
            t_fine_s += omp_get_wtime() - t0;
            uint64_t this_vecs = 0;
            for (int t = 0; t < nthreads; ++t) {
                this_vecs          += q_tc[t].vecs;
                thread_acc[t].vecs += q_tc[t].vecs;
                thread_acc[t].cyc  += q_tc[t].cyc;
                thread_acc[t].ins  += q_tc[t].ins;
                thread_acc[t].l1m  += q_tc[t].l1m;
                thread_acc[t].l1a  += q_tc[t].l1a;
                thread_acc[t].llcm += q_tc[t].llcm;
                thread_acc[t].llca += q_tc[t].llca;
                thread_acc[t].brm  += q_tc[t].brm;
                thread_acc[t].bri  += q_tc[t].bri;
                if (q_tc[t].ok) thread_acc[t].ok = true;
            }
            proc_vecs += (double)this_vecs;
        }

        // ── Merge ─────────────────────────────────────────────────────────────
        std::priority_queue<std::pair<float,uint32_t>> result;
        for (auto& h : thread_heaps) {
            while (!h.empty()) {
                auto top=h.top(); h.pop();
                if (result.size()<k) result.push(top);
                else if (top.first<result.top().first){result.pop();result.push(top);}
            }
        }

        if (measure) {
            gettimeofday(&wall_t1, NULL);
            local_latencies[prof_i] = tv_diff_us(wall_t0, wall_t1);
            std::set<uint32_t> gtset;
            for (int j=0;j<(int)k;++j) gtset.insert((uint32_t)gt_all[j+qi*(int)gt_d]);
            size_t acc=0;
            while (!result.empty()){if(gtset.count(result.top().second))++acc;result.pop();}
            local_recalls[prof_i] = (float)acc/k;
            total_recall += local_recalls[prof_i];
        }
    }

    // ── Final MPI_Gather ──────────────────────────────────────────────────────
    const int total_profiled = size * PROFILE_QUERIES;
    std::vector<float>   all_recalls;
    std::vector<int64_t> all_latencies;
    if (rank==0){ all_recalls.resize(total_profiled); all_latencies.resize(total_profiled); }

    double gather_t0 = MPI_Wtime();
    MPI_Gather(local_recalls  .data(), PROFILE_QUERIES, MPI_FLOAT,
               all_recalls    .data(), PROFILE_QUERIES, MPI_FLOAT,     0, MPI_COMM_WORLD);
    MPI_Gather(local_latencies.data(), PROFILE_QUERIES, MPI_LONG_LONG,
               all_latencies  .data(), PROFILE_QUERIES, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    double gather_s = MPI_Wtime() - gather_t0;

    // ── Cross-process aggregates ──────────────────────────────────────────────
    double fine_max=0,fine_min=0,fine_sum=0;
    double vecs_max=0,vecs_min=0,vecs_sum=0;
    double coarse_max=0,coarse_min=0,coarse_sum=0;
    double gather_max=0;
    MPI_Reduce(&t_fine_s,  &fine_max,   1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&t_fine_s,  &fine_min,   1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
    MPI_Reduce(&t_fine_s,  &fine_sum,   1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&t_coarse_s,&coarse_max, 1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&t_coarse_s,&coarse_min, 1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
    MPI_Reduce(&t_coarse_s,&coarse_sum, 1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&proc_vecs, &vecs_max,   1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&proc_vecs, &vecs_min,   1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
    MPI_Reduce(&proc_vecs, &vecs_sum,   1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&gather_s,  &gather_max, 1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

    // ── Report ────────────────────────────────────────────────────────────────
    if (rank == 0) {
        const double N=PROFILE_QUERIES, NQ=N*size, us=1e6;

        double avg_recall=0, avg_lat_us=0;
        for (int i=0;i<total_profiled;++i){
            avg_recall += all_recalls[i];
            avg_lat_us += (double)all_latencies[i];
        }
        avg_recall /= NQ; avg_lat_us /= NQ;

        const bool have_perf = thread_acc[0].ok;

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "\n════════════════════════════════════════════════════\n";
        std::cout << " IVF_MPI_OMP_PLAN2_V2 (reordered) — perf profile\n";
        std::cout << " mpi_procs=" << size
                  << "  omp_threads=" << nthreads
                  << "  nprobe=" << nprobe << "\n";
        std::cout << "════════════════════════════════════════════════════\n";
        std::cout << "recall@10        : " << avg_recall << "\n";
        std::cout << "avg latency (us) : " << std::setprecision(2) << avg_lat_us << "\n\n";

        std::cout << std::setprecision(2);
        std::cout << "── Phase timings (avg/query, rank 0) ──\n";
        std::cout << "  coarse (single-thread) : " << t_coarse_s/N*us << " us\n";
        std::cout << "  fine   (OMP " << nthreads << " threads, reordered) : "
                  << t_fine_s/N*us << " us\n";
        std::cout << "  total compute          : " << (t_coarse_s+t_fine_s)/N*us << " us\n\n";

        std::cout << "── MPI communication ──\n";
        std::cout << "  gather (once/batch, " << total_profiled << " queries) : "
                  << gather_max*1000.0 << " ms\n";
        std::cout << "  gather amortized/query : " << gather_max/NQ*us << " us\n\n";

        std::cout << "── Per-OMP-thread breakdown — rank 0 (avg/query) ──\n";

        if (have_perf) {
            std::cout << std::setw(6) << "thd" << std::setw(8) << "vecs"
                      << std::setw(7) << "IPC" << std::setw(9) << "cyc/vec";
            if (have_l1)  std::cout << std::setw(10) << "L1miss%";
            if (have_llc) std::cout << std::setw(10) << "LLCmiss%" << std::setw(9) << "LLCMPKI";
            if (have_br)  std::cout << std::setw(9)  << "br-miss%";
            std::cout << "\n";
            print_sep(have_l1 && have_llc && have_br ? 77 : 50);

            uint64_t tot_vecs=0,tot_cyc=0,tot_ins=0;
            uint64_t tot_l1m=0,tot_l1a=0,tot_llcm=0,tot_llca=0,tot_brm=0,tot_bri=0;
            double max_vecs=0;

            for (int t = 0; t < nthreads; ++t) {
                const auto& a = thread_acc[t];
                double vecs=a.vecs/N, cyc=a.cyc/N, ins=a.ins/N;
                double ipc=(cyc>0)?ins/cyc:0.0, cpv=(vecs>0)?cyc/vecs:0.0;
                max_vecs = std::max(max_vecs, vecs);
                std::cout << std::setw(6) << t
                          << std::setw(8) << std::setprecision(1) << vecs
                          << std::setw(7) << std::setprecision(2) << ipc
                          << std::setw(9) << std::setprecision(2) << cpv;
                if (have_l1) {
                    double r=rate(a.l1m,a.l1a);
                    if(r>=0) std::cout<<std::setw(9)<<std::setprecision(1)<<r<<"%";
                    else     std::cout<<std::setw(10)<<"n/a";
                }
                if (have_llc) {
                    double r=rate(a.llcm,a.llca);
                    double mpki=(ins>0)?(a.llcm/N)/(ins/1000.0):0.0;
                    if(r>=0) std::cout<<std::setw(9)<<std::setprecision(1)<<r<<"%"
                                      <<std::setw(9)<<std::setprecision(2)<<mpki;
                    else     std::cout<<std::setw(10)<<"n/a"<<std::setw(9)<<"n/a";
                }
                if (have_br) {
                    double r=rate(a.brm,a.bri);
                    if(r>=0) std::cout<<std::setw(8)<<std::setprecision(1)<<r<<"%";
                    else     std::cout<<std::setw(9)<<"n/a";
                }
                std::cout << "\n";
                tot_vecs+=a.vecs; tot_cyc+=a.cyc; tot_ins+=a.ins;
                tot_l1m+=a.l1m;  tot_l1a+=a.l1a;
                tot_llcm+=a.llcm; tot_llca+=a.llca;
                tot_brm+=a.brm;  tot_bri+=a.bri;
            }
            print_sep(have_l1 && have_llc && have_br ? 77 : 50);

            double tv=tot_vecs/N,tc=tot_cyc/N,ti=tot_ins/N;
            std::cout << std::setw(6) << "all"
                      << std::setw(8) << std::setprecision(1) << tv
                      << std::setw(7) << std::setprecision(2) << (tc>0?ti/tc:0.0)
                      << std::setw(9) << std::setprecision(2) << (tv>0?tc/tv:0.0);
            if(have_l1){ double r=rate(tot_l1m,tot_l1a); if(r>=0) std::cout<<std::setw(9)<<std::setprecision(1)<<r<<"%"; else std::cout<<std::setw(10)<<"n/a"; }
            if(have_llc){ double r=rate(tot_llcm,tot_llca); double mpki=(ti>0)?(tot_llcm/N)/(ti/1000.0):0.0; if(r>=0) std::cout<<std::setw(9)<<std::setprecision(1)<<r<<"%"<<std::setw(9)<<std::setprecision(2)<<mpki; else std::cout<<std::setw(10)<<"n/a"<<std::setw(9)<<"n/a"; }
            if(have_br){ double r=rate(tot_brm,tot_bri); if(r>=0) std::cout<<std::setw(8)<<std::setprecision(1)<<r<<"%"; else std::cout<<std::setw(9)<<"n/a"; }
            std::cout << "\n\n";

            std::cout << "── Cache summary (rank 0, all threads combined) ──\n";
            if (have_l1) {
                double r=rate(tot_l1m,tot_l1a);
                double mpki=(tot_ins>0)?(tot_l1m/N)/((tot_ins/N)/1000.0):0.0;
                std::cout << "  L1-dcache miss rate : ";
                if(r>=0) std::cout<<std::setprecision(2)<<r<<"%  ("<<std::setprecision(1)<<tot_l1m/N<<" misses / "<<tot_l1a/N<<" accesses/query)  MPKI="<<std::setprecision(2)<<mpki<<"\n";
                else     std::cout<<"n/a\n";
            }
            if (have_llc) {
                double r=rate(tot_llcm,tot_llca);
                double mpki=(tot_ins>0)?(tot_llcm/N)/((tot_ins/N)/1000.0):0.0;
                std::cout << "  LLC miss rate       : ";
                if(r>=0) std::cout<<std::setprecision(2)<<r<<"%  ("<<std::setprecision(1)<<tot_llcm/N<<" misses / "<<tot_llca/N<<" accesses/query)  MPKI="<<std::setprecision(2)<<mpki<<"\n";
                else     std::cout<<"n/a\n";
            }
            if (have_br) {
                double r=rate(tot_brm,tot_bri);
                std::cout << "  Branch miss rate    : ";
                if(r>=0) std::cout<<std::setprecision(2)<<r<<"%  ("<<std::setprecision(1)<<tot_brm/N<<" mispred / "<<tot_bri/N<<" branches/query)\n";
                else     std::cout<<"n/a\n";
            }
            std::cout << "\n";

            double mean_vecs = tv / nthreads;
            std::cout << "── Intra-process OMP load balance (rank 0) ──\n";
            std::cout << "  mean vecs/thread : " << std::setprecision(1) << mean_vecs << "\n";
            std::cout << "  max  vecs/thread : " << max_vecs << "\n";
            std::cout << "  imbalance        : " << std::setprecision(2)
                      << ((mean_vecs>0)?(max_vecs/mean_vecs-1.0)*100.0:0.0) << "%\n\n";
        } else {
            std::cout << "  [perf blocked — showing vector counts only]\n";
            std::cout << std::setw(8) << "thread" << std::setw(12) << "vecs/query\n";
            print_sep(20);
            double max_v=0, mean_v=0;
            for (int t=0;t<nthreads;++t){
                double v=thread_acc[t].vecs/N;
                std::cout<<std::setw(8)<<t<<std::setw(12)<<std::setprecision(1)<<v<<"\n";
                max_v=std::max(max_v,v); mean_v+=v;
            }
            mean_v/=nthreads;
            std::cout<<"\n── Intra-process OMP load balance ──\n";
            std::cout<<"  mean="<<std::setprecision(1)<<mean_v<<"  max="<<max_v
                     <<"  imbalance="<<std::setprecision(2)
                     <<((mean_v>0)?(max_v/mean_v-1.0)*100.0:0.0)<<"%\n\n";
        }

        std::cout << "── Cross-process MPI load balance ──\n";
        std::cout << "  vecs/proc   avg=" << std::setprecision(1) << vecs_sum/size/N
                  << "  max=" << vecs_max/N << "  min=" << vecs_min/N
                  << "  imbal=" << std::setprecision(2)
                  << ((vecs_max>0)?(vecs_max-vecs_min)/vecs_max*100.0:0.0) << "%\n";
        std::cout << "  coarse/proc avg=" << std::setprecision(2) << coarse_sum/size/N*us
                  << "us  max=" << coarse_max/N*us << "us  min=" << coarse_min/N*us << "us\n";
        std::cout << "  fine/proc   avg=" << fine_sum/size/N*us
                  << "us  max=" << fine_max/N*us << "us  min=" << fine_min/N*us
                  << "us  imbal=" << std::setprecision(2)
                  << ((fine_max>0)?(fine_max-fine_min)/fine_max*100.0:0.0) << "%\n";
        std::cout << "════════════════════════════════════════════════════\n";
    }

    delete[] queries_all; delete[] gt_all; delete[] base;
    MPI_Finalize();
    return 0;
}
