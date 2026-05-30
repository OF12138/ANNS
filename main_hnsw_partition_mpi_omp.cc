// =============================================================================
// main_hnsw_partition_mpi_omp.cc -- P-partition HNSW, MPI x OMP
//
// Approach: base is randomly split into NUM_PARTS equal partitions (fixed seed).
// One HNSW is built per partition.  At query time, all P MPI processes
// collaborate: process r searches its NUM_PARTS/P partition slice with T OMP
// threads (multi-entry), then MPI_Gather merges the global top-k at rank 0.
//
// Key contrast with IVF+HNSW:
//   - No coarse quantization -- every partition is always searched.
//   - Each process loads only its own partitions (~base_number/NUM_PARTS vectors)
//     instead of the full index, so memory scales as 1/NP per process.
//
// Compile:
//   mpicxx main_hnsw_partition_mpi_omp.cc -o main_hnsw_partition_mpi_omp \
//          -O2 -std=c++11 -fopenmp -lpthread -lm
// =============================================================================

#define NUM_PARTS            8
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

#include "ARM/Alg_parallel/hnsw_partition_mpi_omp.h"


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
        std::cerr << "[config] P-partition HNSW  MPI(partition-split) x OMP(multi-entry)\n";
        std::cerr << "[config] mpi_procs="      << size
                  << "  omp_threads="           << nthreads
                  << "  total_workers="         << size * nthreads << "\n";
        std::cerr << "[config] num_parts="      << NUM_PARTS       << "\n";
        std::cerr << "[config] hnsw_M="         << HNSW_M
                  << "  ef_construction="       << HNSW_EF_CONSTRUCTION
                  << "  ef_search="             << HNSW_EF_SEARCH  << "\n";
        std::cerr << "========================================\n";
    }

    // ── Load query + ground truth (small; all processes load these) ────────────
    const std::string data_path = "/anndata/";
    size_t test_number = 0, test_gt_d = 0, vecdim = 0;

    float* test_query = LoadData<float>(data_path + "DEEP100K.query.fbin",
                                        test_number, vecdim);
    int*   test_gt    = LoadData<int>  (data_path + "DEEP100K.gt.query.100k.top100.bin",
                                        test_number, test_gt_d);
    test_number = 2000;
    const size_t k = 10;

    if (rank == 0)
        std::cerr << "[data] queries=" << test_number << "  dim=" << vecdim << "\n";

    // ── Compute this process's partition slice ─────────────────────────────────
    const int np_int = NUM_PARTS;
    const int chunk  = np_int / size;
    const int rem    = np_int % size;
    const int lo_p   = rank * chunk + std::min(rank, rem);
    const int hi_p   = lo_p + chunk + (rank < rem ? 1 : 0);

    // ── Build or load assigned partitions ─────────────────────────────────────
    PartitionedHNSWIndex idx;
    char cache_dir[256];
    snprintf(cache_dir, sizeof(cache_dir),
             "files/hnsw_parts_N%d_M%d_ef%d",
             NUM_PARTS, HNSW_M, HNSW_EF_CONSTRUCTION);

    struct timeval tb0, tb1;
    gettimeofday(&tb0, NULL);

    bool loaded = hnsw_partition_load(idx, cache_dir, vecdim,
                                      HNSW_M, HNSW_EF_CONSTRUCTION,
                                      lo_p, hi_p);
    if (loaded) {
        gettimeofday(&tb1, NULL);
        if (rank == 0)
            std::cerr << "[build] loaded from cache: "
                      << tv_diff_us(tb0, tb1) / 1000 << " ms"
                      << "  (parts " << lo_p << ".." << hi_p - 1
                      << " of " << NUM_PARTS << ")\n";
    } else {
        // Cache miss: rank 0 builds all NUM_PARTS partitions and saves.
        // Only rank 0 loads base (avoids holding 38 MB on every process).
        if (rank == 0) {
            size_t base_number = 0, base_vecdim = 0;
            float* base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin",
                                          base_number, base_vecdim);
            std::cerr << "[data] base=" << base_number << "\n";
            std::cerr << "[build] building " << NUM_PARTS << " partition HNSWs...\n";

            PartitionedHNSWIndex idx_build;
            hnsw_partition_build(idx_build, base, base_number, vecdim,
                                 NUM_PARTS, HNSW_M, HNSW_EF_CONSTRUCTION, /*seed=*/42);
            delete[] base;

            gettimeofday(&tb1, NULL);
            std::cerr << "[build] done: "
                      << tv_diff_us(tb0, tb1) / 1000 << " ms\n";
            std::cerr << "[build] saving to " << cache_dir << " ...\n";
            hnsw_partition_save(idx_build, cache_dir);
            hnsw_partition_free(idx_build);
            std::cerr << "[build] saved.\n";
        }
        // Barrier: ensure rank 0 finishes saving before others attempt to load
        MPI_Barrier(MPI_COMM_WORLD);

        gettimeofday(&tb0, NULL);
        hnsw_partition_load(idx, cache_dir, vecdim,
                            HNSW_M, HNSW_EF_CONSTRUCTION,
                            lo_p, hi_p);
        gettimeofday(&tb1, NULL);
        if (rank == 0)
            std::cerr << "[build] all processes loaded assigned partitions: "
                      << tv_diff_us(tb0, tb1) / 1000 << " ms\n";
    }

    // ── Warm-up ───────────────────────────────────────────────────────────────
    if (rank == 0)
        std::cerr << "[warmup] running " << test_number << " queries...\n";
    for (size_t i = 0; i < test_number; ++i) {
        hnsw_partition_mpi_omp_search(
            idx, test_query + i * vecdim,
            k, HNSW_EF_SEARCH, rank, size, nthreads);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // ── Measured run ──────────────────────────────────────────────────────────
    std::vector<float>   recalls  (test_number, 0.0f);
    std::vector<int64_t> latencies(test_number, 0LL);
    HNSWPartitionTimings tm;

    struct timeval batch_t0, batch_t1;
    MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday(&batch_t0, NULL);

    for (size_t i = 0; i < test_number; ++i) {
        const float* q = test_query + i * vecdim;

        struct timeval t0, t1;
        gettimeofday(&t0, NULL);

        auto res = hnsw_partition_mpi_omp_search(
            idx, q, k, HNSW_EF_SEARCH, rank, size, nthreads, &tm);

        gettimeofday(&t1, NULL);

        if (rank == 0) {
            latencies[i] = tv_diff_us(t0, t1);

            std::set<uint32_t> gtset;
            for (size_t j = 0; j < k; ++j)
                gtset.insert(static_cast<uint32_t>(
                    test_gt[j + i * test_gt_d]));
            std::set<uint32_t> found;
            while (!res.empty()) {
                found.insert(res.top().second);
                res.pop();
            }
            size_t acc = 0;
            for (uint32_t id : found)
                if (gtset.count(id)) ++acc;
            recalls[i] = static_cast<float>(acc) / k;
        }
    }

    gettimeofday(&batch_t1, NULL);
    double my_batch_s  = tv_diff_us(batch_t0, batch_t1) / 1e6;
    double my_search_s = tm.t_search_s;

    // ── Cross-process timing stats ────────────────────────────────────────────
    double batch_max = 0, search_max = 0, search_min = 0, search_sum = 0;
    MPI_Reduce(&my_batch_s,  &batch_max,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&my_search_s, &search_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&my_search_s, &search_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&my_search_s, &search_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

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
        std::cerr << "  search : " << search_sum / size / N * us << " us  avg"
                  << "  (OMP " << nthreads << " threads/proc, multi-entry)\n";

        std::cerr << "\n[load balance across " << size << " processes]\n";
        std::cerr << "  search max/min : "
                  << search_max / N * us << " / " << search_min / N * us << " us\n";
        std::cerr << "  imbalance      : "
                  << 100.0 * (search_max - search_min) / search_max << "%\n";

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

    hnsw_partition_free(idx);
    delete[] test_query;
    delete[] test_gt;
    MPI_Finalize();
    return 0;
}
