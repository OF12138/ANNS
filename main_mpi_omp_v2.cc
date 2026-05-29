// =============================================================================
// main_mpi_omp_v2.cc — Plan II with data reordering optimization
//
// Identical to main_mpi_omp.cc except:
//   IVF_REORDER = 1   → index built with reordered_base (cluster-contiguous)
//   Fine scan uses ivf_omp_search_query_reorder / _timed_reorder
//     → reads reordered_base[j*d] (sequential) instead of base[orig*d] (random)
//     → HW prefetcher effective → LLC miss rate drops significantly
//
// Compile:
//   mpicxx main_mpi_omp_v2.cc -o main_mpi_omp_v2 -O2 -std=c++11 -fopenmp -lm
// =============================================================================

#define IVF_NLIST   1024
#define IVF_NPROBE  16
#define IVF_REORDER 1           // ← reorder enabled

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

    const int nthreads = omp_get_max_threads();

    if (rank == 0) {
        std::cerr << "========================================\n";
        std::cerr << "[config] IVF_MPI_OMP_V2  (reordered base)\n";
        std::cerr << "[config] mpi_procs="  << size
                  << "  omp_threads="       << nthreads
                  << "  total_workers="     << size * nthreads << "\n";
        std::cerr << "[config] ivf_nlist="  << IVF_NLIST
                  << "  ivf_nprobe="        << IVF_NPROBE
                  << "  reorder=1\n";
        std::cerr << "========================================\n";
    }

    // ── Load data ─────────────────────────────────────────────────────────────
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

    const int slice_size = static_cast<int>(test_number) / size;
    const int lo_q       = rank * slice_size;
    const int hi_q       = lo_q + slice_size;

    if (rank == 0)
        std::cerr << "[config] queries/proc=" << slice_size << "\n";

    // ── Build / load IVFIndex (reorder=1, separate cache file) ───────────────
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
                          IVF_NLIST, 25, true);   // reorder=true
                gettimeofday(&tb1, NULL);
                std::cerr << "[build] IVFIndex built (reordered): "
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
        ivf_omp_search_query_reorder(ivf_index, base,
                                     test_query + static_cast<size_t>(i) * vecdim,
                                     k, IVF_NPROBE, nthreads);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // ── Measured run ──────────────────────────────────────────────────────────
    std::vector<float>   local_recalls  (slice_size, 0.0f);
    std::vector<int64_t> local_latencies(slice_size, 0LL);

    HybridTimings tm;

    struct timeval batch_t0, batch_t1;
    MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday(&batch_t0, NULL);

    for (int i = lo_q; i < hi_q; ++i) {
        struct timeval t0, t1;
        gettimeofday(&t0, NULL);

        auto res = ivf_omp_search_query_timed_reorder(
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

    // ── Gather ────────────────────────────────────────────────────────────────
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

    // ── Cross-process stats ───────────────────────────────────────────────────
    double coarse_local = tm.t_coarse_s, fine_local = tm.t_fine_s;
    double batch_max=0, coarse_max=0, coarse_min=0, coarse_sum=0;
    double fine_max=0, fine_min=0, fine_sum=0;

    MPI_Reduce(&my_batch_s,   &batch_max,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&coarse_local, &coarse_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&coarse_local, &coarse_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&coarse_local, &coarse_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&fine_local,   &fine_max,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&fine_local,   &fine_min,   1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&fine_local,   &fine_sum,   1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // ── Report ────────────────────────────────────────────────────────────────
    if (rank == 0) {
        double avg_recall = 0.0, avg_lat_us = 0.0;
        for (size_t i = 0; i < test_number; ++i) {
            avg_recall += all_recalls[i];
            avg_lat_us += static_cast<double>(all_latencies[i]);
        }
        avg_recall /= test_number;
        avg_lat_us /= test_number;

        const double N  = static_cast<double>(slice_size);
        const double us = 1e6;

        std::cerr << std::fixed << std::setprecision(2);
        std::cerr << "\n[phase breakdown — avg per query, per-process view]\n";
        std::cerr << "  coarse : " << coarse_sum / size / N * us << " us  avg\n";
        std::cerr << "  fine   : " << fine_sum   / size / N * us << " us  avg  "
                  << "(reordered, OMP " << nthreads << " threads)\n";

        std::cerr << "\n[load balance across " << size << " processes]\n";
        std::cerr << "  coarse  max/min : "
                  << coarse_max / N * us << " / " << coarse_min / N * us << " us\n";
        std::cerr << "  fine    max/min : "
                  << fine_max / N * us << " / " << fine_min / N * us << " us\n";
        std::cerr << "  fine imbalance  : "
                  << (fine_max - fine_min) / N * us << " us"
                  << "  (" << 100.0 * (fine_max - fine_min) / fine_max << "%)\n";

        std::cerr << "\n[throughput]\n";
        std::cerr << "  batch wall time : " << batch_max * 1000.0 << " ms\n";
        std::cerr << "  throughput      : "
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
