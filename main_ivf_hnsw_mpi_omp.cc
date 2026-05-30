// =============================================================================
// main_ivf_hnsw_mpi_omp.cc -- IVF+HNSW, MPI (nprobe-split) x OMP (multi-entry)
//
// MPI: P processes split the nprobe clusters.  All P processes collaborate on
//      each query, reducing per-query latency.
// OMP: T threads per process run independent HNSW beam searches from different
//      entry points within each assigned cluster.
//
// Compile:
//   mpicxx main_ivf_hnsw_mpi_omp.cc -o main_ivf_hnsw_mpi_omp \
//          -O2 -std=c++11 -fopenmp -lpthread -lm
// =============================================================================

#define IVF_NLIST            1024
#define IVF_NPROBE           16
#define HNSW_M               16
#define HNSW_EF_CONSTRUCTION 200
#define HNSW_EF_SEARCH       50

#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <iomanip>
#include <sys/time.h>

#include "ARM/Alg_parallel/ivf_hnsw_mpi_omp.h"


template<typename T>
T* LoadData(const std::string& path, size_t& n, size_t& d)
{
    std::ifstream fin(path, std::ios::binary);
    fin.read(reinterpret_cast<char*>(&n), 4);
    fin.read(reinterpret_cast<char*>(&d), 4);
    T* data = new T[n * d];
    for (size_t i = 0; i < n; ++i)
        fin.read(reinterpret_cast<char*>(data) + i * d * sizeof(T),
                 d * sizeof(T));
    fin.close();
    return data;
}

static inline int64_t tv_diff_us(const struct timeval& a,
                                  const struct timeval& b)
{
    return (b.tv_sec * 1000000LL + b.tv_usec)
         - (a.tv_sec * 1000000LL + a.tv_usec);
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
        std::cerr << "[config] IVF+HNSW  MPI(nprobe-split) x OMP(multi-entry)\n";
        std::cerr << "[config] mpi_procs="   << size
                  << "  omp_threads="        << nthreads
                  << "  total_workers="      << size * nthreads << "\n";
        std::cerr << "[config] ivf_nlist="   << IVF_NLIST
                  << "  ivf_nprobe="         << IVF_NPROBE      << "\n";
        std::cerr << "[config] hnsw_M="      << HNSW_M
                  << "  ef_construction="    << HNSW_EF_CONSTRUCTION
                  << "  ef_search="          << HNSW_EF_SEARCH   << "\n";
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

    if (rank == 0)
        std::cerr << "[data] base=" << base_number
                  << "  queries=" << test_number
                  << "  dim=" << vecdim << "\n";

    // ── Build or load IVF+HNSW ────────────────────────────────────────────────
    IVFHNSWIndex idx;
    char cache_dir[256];
    snprintf(cache_dir, sizeof(cache_dir),
             "files/ivf_hnsw_nlist%d_M%d_ef%d",
             IVF_NLIST, HNSW_M, HNSW_EF_CONSTRUCTION);

    struct timeval tb0, tb1;
    gettimeofday(&tb0, NULL);

    bool loaded = ivf_hnsw_load(idx, cache_dir, HNSW_M, HNSW_EF_CONSTRUCTION);
    if (loaded) {
        gettimeofday(&tb1, NULL);
        if (rank == 0)
            std::cerr << "[build] loaded from cache: "
                      << tv_diff_us(tb0, tb1) / 1000 << " ms\n";
    } else {
        if (rank == 0) std::cerr << "[build] building IVF+HNSW...\n";
        ivf_hnsw_build(idx, base, base_number, vecdim,
                       IVF_NLIST, 25, HNSW_M, HNSW_EF_CONSTRUCTION);
        gettimeofday(&tb1, NULL);
        if (rank == 0) {
            std::cerr << "[build] done: "
                      << tv_diff_us(tb0, tb1) / 1000 << " ms\n";
            std::cerr << "[build] saving to " << cache_dir << " ...\n";
            ivf_hnsw_save(idx, cache_dir);
            std::cerr << "[build] saved.\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank != 0) {
            ivf_hnsw_free(idx);
            ivf_hnsw_load(idx, cache_dir, HNSW_M, HNSW_EF_CONSTRUCTION);
        }
    }

    // ── Warm-up ───────────────────────────────────────────────────────────────
    if (rank == 0) std::cerr << "[warmup] running " << test_number << " queries...\n";
    for (size_t i = 0; i < test_number; ++i) {
        ivf_hnsw_mpi_omp_search_query(
            idx, test_query + i * vecdim,
            k, IVF_NPROBE, HNSW_EF_SEARCH,
            rank, size, nthreads);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // ── Measured run ──────────────────────────────────────────────────────────
    std::vector<float>   recalls  (test_number, 0.0f);
    std::vector<int64_t> latencies(test_number, 0LL);
    IVFHNSWMPITimings tm;

    struct timeval batch_t0, batch_t1;
    MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday(&batch_t0, NULL);

    for (size_t i = 0; i < test_number; ++i) {
        const float* q = test_query + i * vecdim;

        struct timeval t0, t1;
        gettimeofday(&t0, NULL);

        auto res = ivf_hnsw_mpi_omp_search_query(
            idx, q, k, IVF_NPROBE, HNSW_EF_SEARCH,
            rank, size, nthreads, &tm);

        gettimeofday(&t1, NULL);

        if (rank == 0) {
            latencies[i] = tv_diff_us(t0, t1);

            std::set<uint32_t> gtset;
            for (size_t j = 0; j < k; ++j)
                gtset.insert(static_cast<uint32_t>(
                    test_gt[j + i * test_gt_d]));
            size_t acc = 0;
            while (!res.empty()) {
                if (gtset.count(res.top().second)) ++acc;
                res.pop();
            }
            recalls[i] = static_cast<float>(acc) / k;
        }
    }

    gettimeofday(&batch_t1, NULL);
    double my_batch_s = tv_diff_us(batch_t0, batch_t1) / 1e6;

    // ── Cross-process stats ───────────────────────────────────────────────────
    double coarse_local = tm.t_coarse_s;
    double fine_local   = tm.t_fine_s;
    double batch_max = 0, coarse_max = 0, coarse_min = 0;
    double fine_max = 0, fine_min = 0, fine_sum = 0, coarse_sum = 0;

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
            avg_recall  += recalls[i];
            avg_lat_us  += latencies[i];
        }
        avg_recall /= test_number;
        avg_lat_us /= test_number;

        const double N  = static_cast<double>(test_number);
        const double us = 1e6;

        std::cerr << std::fixed << std::setprecision(2);
        std::cerr << "\n[phase breakdown -- avg per query]\n";
        std::cerr << "  coarse : " << coarse_sum / size / N * us << " us  avg\n";
        std::cerr << "  fine   : " << fine_sum   / size / N * us << " us  avg"
                  << "  (OMP " << nthreads << " threads/proc, multi-entry)\n";

        std::cerr << "\n[load balance across " << size << " processes]\n";
        std::cerr << "  coarse  max/min : "
                  << coarse_max / N * us << " / " << coarse_min / N * us << " us\n";
        std::cerr << "  fine    max/min : "
                  << fine_max / N * us << " / " << fine_min / N * us << " us\n";
        std::cerr << "  fine imbalance  : "
                  << 100.0 * (fine_max - fine_min) / fine_max << "%\n";

        double throughput_qs = N / batch_max;
        double amortized_us  = 1000000.0 / throughput_qs;

        std::cerr << "\n[throughput]\n";
        std::cerr << "  batch wall time   : " << batch_max * 1000.0 << " ms\n";
        std::cerr << "  throughput        : " << throughput_qs << " queries/sec\n";
        std::cerr << "  amortized latency : " << amortized_us << " us/query\n";

        std::cout << std::fixed << std::setprecision(5);
        std::cout << "average recall: "         << avg_recall   << "\n";
        std::cout << "average latency (us): "   << avg_lat_us   << "\n";
        std::cout << "amortized latency (us): " << amortized_us << "\n";
        std::cout << "throughput (q/s): "       << throughput_qs << "\n";
    }

    ivf_hnsw_free(idx);
    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    MPI_Finalize();
    return 0;
}
