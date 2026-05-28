// =============================================================================
// main_mpi_omp.cc — Hybrid MPI × OMP benchmark harness (Plan II)
//
// Parallelization:
//   MPI:  rank r processes queries[ r*Q/P .. (r+1)*Q/P )  — no per-query Bcast
//   OMP:  T threads parallelize the fine scan within each query
//
// Each process independently loads data from NFS and runs its query slice.
// Results are collected to rank 0 with a single MPI_Gather at the end.
//
// Key parameters (change to run experiments):
//   IVF_NPROBE       — clusters probed per query (recall-latency knob)
//   OMP_NUM_THREADS  — set in qsub_mpi_omp.sh (env var read by omp_get_max_threads)
//   PBS nodes/ppn/NP — set in qsub_mpi_omp.sh
//
// Compile:
//   mpicxx main_mpi_omp.cc -o main_mpi_omp -O2 -std=c++11 -fopenmp -lm
// =============================================================================

#define IVF_NLIST   1024
#define IVF_NPROBE  16
#define IVF_REORDER 0

#include <mpi.h>
#include <omp.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <cstring>
#include <cfloat>
#include <cstdint>
#include <algorithm>
#include <iomanip>
#include <sys/time.h>
#include "ARM/Alg_parallel/ivf_flat_simd_mpi_omp.h"
#include "ARM/Alg_parallel/ivf_flat_simd.h"


template<typename T>
T* LoadData(const std::string& path, size_t& n, size_t& d)
{
    std::ifstream fin(path, std::ios::binary);
    fin.read(reinterpret_cast<char*>(&n), 4);
    fin.read(reinterpret_cast<char*>(&d), 4);
    T* data = new T[n * d];
    for (int i = 0; i < static_cast<int>(n); ++i)
        fin.read(reinterpret_cast<char*>(data) + i * d * sizeof(T), d * sizeof(T));
    fin.close();
    return data;
}

static inline int64_t tv_diff_us(const struct timeval& a, const struct timeval& b)
{
    return (b.tv_sec * 1000000LL + b.tv_usec) - (a.tv_sec * 1000000LL + a.tv_usec);
}


int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int nthreads = omp_get_max_threads();   // reads OMP_NUM_THREADS env var

    // ── Config printout ───────────────────────────────────────────────────────
    if (rank == 0) {
        std::cerr << "========================================\n";
        std::cerr << "[config] IVF_MPI_OMP  (query-batch × fine-scan hybrid)\n";
        std::cerr << "[config] mpi_procs="   << size
                  << "  omp_threads="        << nthreads
                  << "  total_workers="      << size * nthreads << "\n";
        std::cerr << "[config] ivf_nlist="   << IVF_NLIST
                  << "  ivf_nprobe="         << IVF_NPROBE << "\n";
        std::cerr << "========================================\n";
    }

    // ── Load data (all processes, from shared NFS) ────────────────────────────
    const std::string data_path = "/anndata/";
    size_t test_number = 0, base_number = 0, test_gt_d = 0, vecdim = 0;

    float* test_query = LoadData<float>(data_path + "DEEP100K.query.fbin",
                                        test_number, vecdim);
    int*   test_gt    = LoadData<int>  (data_path + "DEEP100K.gt.query.100k.top100.bin",
                                        test_number, test_gt_d);
    float* base       = LoadData<float>(data_path + "DEEP100K.base.100k.fbin",
                                        base_number, vecdim);
    test_number = 2000;
    const size_t k = 10;

    // ── Query slice for this process ──────────────────────────────────────────
    // 2000 divides evenly for P = 1, 2, 4, 5, 8, 10, 16, 20, 25 — no remainder
    const int slice_size = static_cast<int>(test_number) / size;
    const int lo_q       = rank * slice_size;
    const int hi_q       = lo_q + slice_size;

    if (rank == 0)
        std::cerr << "[config] queries/proc=" << slice_size
                  << "  (rank 0 handles [" << lo_q << ".." << hi_q << "))\n";

    // ── Build / load IVFIndex ─────────────────────────────────────────────────
    IVFIndex ivf_index;
    {
        char cache_path[256];
        snprintf(cache_path, sizeof(cache_path),
                 "files/ivf_nlist%d_reorder%d.bin", IVF_NLIST, IVF_REORDER);

        struct timeval tb0, tb1;
        gettimeofday(&tb0, NULL);
        bool loaded = ivf_load(ivf_index, cache_path);

        if (!loaded) {
            if (rank == 0) {
                ivf_build(ivf_index, base, base_number, vecdim,
                          IVF_NLIST, 25, IVF_REORDER != 0);
                gettimeofday(&tb1, NULL);
                std::cerr << "[build] IVFIndex built: "
                          << tv_diff_us(tb0, tb1) / 1000 << " ms\n";
                ivf_save(ivf_index, cache_path);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank != 0) ivf_load(ivf_index, cache_path);
        } else {
            if (rank == 0) {
                gettimeofday(&tb1, NULL);
                std::cerr << "[build] IVFIndex loaded from cache: "
                          << tv_diff_us(tb0, tb1) / 1000 << " ms\n";
            }
        }
    }

    // ── Warm-up ───────────────────────────────────────────────────────────────
    if (rank == 0) std::cerr << "[warmup] running " << slice_size << " queries...\n";
    for (int i = lo_q; i < hi_q; ++i) {
        ivf_omp_search_query(ivf_index, base,
                             test_query + static_cast<size_t>(i) * vecdim,
                             k, IVF_NPROBE, nthreads);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // ── Measured run ──────────────────────────────────────────────────────────
    std::vector<float>   local_recalls  (slice_size, 0.0f);
    std::vector<int64_t> local_latencies(slice_size, 0LL);

    HybridTimings tm;   // zero-initialised by struct default

    struct timeval batch_t0, batch_t1;
    MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday(&batch_t0, NULL);

    for (int i = lo_q; i < hi_q; ++i) {
        struct timeval t0, t1;
        gettimeofday(&t0, NULL);

        auto res = ivf_omp_search_query_timed(
            ivf_index, base,
            test_query + static_cast<size_t>(i) * vecdim,
            k, IVF_NPROBE, nthreads, &tm);

        gettimeofday(&t1, NULL);
        local_latencies[i - lo_q] = tv_diff_us(t0, t1);

        std::set<uint32_t> gtset;
        for (int j = 0; j < static_cast<int>(k); ++j)
            gtset.insert(static_cast<uint32_t>(
                test_gt[j + i * static_cast<int>(test_gt_d)]));
        size_t acc = 0;
        while (!res.empty()) {
            if (gtset.count(res.top().second)) ++acc;
            res.pop();
        }
        local_recalls[i - lo_q] = static_cast<float>(acc) / static_cast<float>(k);
    }

    gettimeofday(&batch_t1, NULL);
    double my_batch_s = tv_diff_us(batch_t0, batch_t1) / 1e6;

    // ── Gather results to rank 0 ──────────────────────────────────────────────
    // [MPI_Gather] all processes → rank 0: recalls and latencies for their slice
    std::vector<float>   all_recalls;
    std::vector<int64_t> all_latencies;
    if (rank == 0) {
        all_recalls  .resize(test_number);
        all_latencies.resize(test_number);
    }

    MPI_Gather(local_recalls  .data(), slice_size, MPI_FLOAT,
               all_recalls    .data(), slice_size, MPI_FLOAT,
               0, MPI_COMM_WORLD);
    MPI_Gather(local_latencies.data(), slice_size, MPI_LONG_LONG,
               all_latencies  .data(), slice_size, MPI_LONG_LONG,
               0, MPI_COMM_WORLD);

    // ── Cross-process phase timing stats ─────────────────────────────────────
    double coarse_local = tm.t_coarse_s;
    double fine_local   = tm.t_fine_s;
    double batch_max    = 0.0;

    double coarse_max = 0.0, coarse_min = 0.0, coarse_sum = 0.0;
    double fine_max   = 0.0, fine_min   = 0.0, fine_sum   = 0.0;

    MPI_Reduce(&my_batch_s,    &batch_max,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&coarse_local,  &coarse_max,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&coarse_local,  &coarse_min,  1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&coarse_local,  &coarse_sum,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&fine_local,    &fine_max,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&fine_local,    &fine_min,    1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&fine_local,    &fine_sum,    1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // ── Report (rank 0 only) ──────────────────────────────────────────────────
    if (rank == 0) {
        double avg_recall = 0.0, avg_lat_us = 0.0;
        for (size_t i = 0; i < test_number; ++i) {
            avg_recall  += all_recalls  [i];
            avg_lat_us  += static_cast<double>(all_latencies[i]);
        }
        avg_recall  /= test_number;
        avg_lat_us  /= test_number;

        const double N  = static_cast<double>(slice_size);   // queries per process
        const double us = 1e6;

        std::cerr << std::fixed << std::setprecision(2);
        std::cerr << "\n[phase breakdown — avg per query, per-process view]\n";
        std::cerr << "  coarse (centroid rank, single-thread)   : "
                  << coarse_sum / size / N * us << " us  avg across procs\n";
        std::cerr << "  fine   (OMP " << nthreads << "-thread cluster scan)    : "
                  << fine_sum   / size / N * us << " us  avg across procs\n";

        std::cerr << "\n[load balance across " << size << " processes]\n";
        std::cerr << "  coarse  max/min : "
                  << coarse_max / N * us << " / " << coarse_min / N * us << " us\n";
        std::cerr << "  fine    max/min : "
                  << fine_max   / N * us << " / " << fine_min   / N * us << " us\n";
        std::cerr << "  fine imbalance  : "
                  << (fine_max - fine_min) / N * us << " us"
                  << "  (" << 100.0 * (fine_max - fine_min) / fine_max << "%)\n";

        std::cerr << "\n[throughput]\n";
        std::cerr << "  batch wall time (slowest proc) : " << batch_max * 1000.0 << " ms\n";
        std::cerr << "  throughput                     : "
                  << static_cast<double>(test_number) / batch_max << " queries/sec\n";

        std::cout << std::fixed << std::setprecision(5);
        std::cout << "average recall: "       << avg_recall  << "\n";
        std::cout << "average latency (us): " << avg_lat_us  << "\n";
        std::cout << "throughput (q/s): "
                  << static_cast<double>(test_number) / batch_max << "\n";
    }

    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    MPI_Finalize();
    return 0;
}
