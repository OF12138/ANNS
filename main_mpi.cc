// =============================================================================
// main_mpi.cc — MPI benchmark harness for IVF-SIMD ANN search
//
// Parallelization: nprobe partition across MPI processes.
//   Each process independently loads the pre-built IVFIndex from a shared
//   NFS cache file.  Per query: rank 0 broadcasts the query, all processes
//   run coarse independently, each process fine-scans its nprobe/P slice,
//   MPI_Gather collects results, rank 0 merges to top-k.
//
// Phase timing (ivf_mpi_search_query_timed):
//   Uses MPI_Wtime() — a lightweight double-precision wall clock (~50 ns/call).
//   6 calls per query × 50 ns = 300 ns overhead/query → < 0.1% of µs-scale
//   latency, invisible in the reported numbers.
//
//   After the measured run, MPI_Reduce collects min/max/avg of t_fine across
//   all processes to quantify load imbalance.
//
// Compile:  mpicxx main_mpi.cc -o main_mpi -O2 -std=c++11 -lm
// Run:      mpiexec -n P ./main_mpi   (P = 1, 2, 4, 8)
// =============================================================================

// ── Configuration ─────────────────────────────────────────────────────────────
#define IVF_NLIST   1024   // must match cached index
#define IVF_NPROBE  16     // clusters probed per query
#define IVF_REORDER 0      // must match cached index
// ─────────────────────────────────────────────────────────────────────────────

#include <mpi.h>
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
#include "ARM/Alg_parallel/ivf_flat_simd_mpi.h"
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

    // ── Config printout ───────────────────────────────────────────────────────
    if (rank == 0) {
        std::cerr << "========================================\n";
        std::cerr << "[config] IVF_MPI_SIMD  (nprobe partition)\n";
        std::cerr << "[config] mpi_procs="  << size
                  << "  ivf_nlist="  << IVF_NLIST
                  << "  ivf_nprobe=" << IVF_NPROBE << "\n";
        std::cerr << "[config] probes/proc (approx): "
                  << IVF_NPROBE / size << " – "
                  << (IVF_NPROBE + size - 1) / size << "\n";
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

    // ── Warm-up (plain version — no timing accumulation) ─────────────────────
    if (rank == 0) std::cerr << "[warmup] running " << test_number << " queries...\n";
    for (int i = 0; i < static_cast<int>(test_number); ++i) {
        ivf_mpi_search_query(ivf_index, base,
                             test_query + static_cast<size_t>(i) * vecdim,
                             k, IVF_NPROBE, rank, size);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // ── Measured run ──────────────────────────────────────────────────────────
    std::vector<float>   recalls  (test_number, 0.0f);
    std::vector<int64_t> latencies(test_number, 0LL);

    IVFMPITimings tm;   // zero-initialised by struct default

    for (int i = 0; i < static_cast<int>(test_number); ++i) {
        struct timeval t0, t1;
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) gettimeofday(&t0, NULL);

        auto res = ivf_mpi_search_query_timed(
            ivf_index, base,
            test_query + static_cast<size_t>(i) * vecdim,
            k, IVF_NPROBE, rank, size, &tm);

        if (rank == 0) {
            gettimeofday(&t1, NULL);
            latencies[i] = tv_diff_us(t0, t1);

            std::set<uint32_t> gtset;
            for (int j = 0; j < static_cast<int>(k); ++j)
                gtset.insert(static_cast<uint32_t>(
                    test_gt[j + i * static_cast<int>(test_gt_d)]));
            size_t acc = 0;
            while (!res.empty()) {
                if (gtset.count(res.top().second)) ++acc;
                res.pop();
            }
            recalls[i] = static_cast<float>(acc) / static_cast<float>(k);
        }
    }

    // ── Cross-process fine-scan timing: collect min / max / sum ──────────────
    // Each process has tm.t_fine_s summed over test_number queries.
    // MPI_Reduce to rank 0 for load-imbalance reporting.
    double fine_sum_local = tm.t_fine_s;
    double fine_max = 0.0, fine_min = 0.0, fine_sum = 0.0;
    MPI_Reduce(&fine_sum_local, &fine_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&fine_sum_local, &fine_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&fine_sum_local, &fine_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // ── Report (rank 0 only) ──────────────────────────────────────────────────
    if (rank == 0) {
        double avg_recall = 0.0, avg_lat_us = 0.0;
        for (int i = 0; i < static_cast<int>(test_number); ++i) {
            avg_recall  += recalls  [i];
            avg_lat_us  += latencies[i];
        }
        avg_recall  /= test_number;
        avg_lat_us  /= test_number;

        const double N = static_cast<double>(test_number);
        // Convert seconds → microseconds for display
        const double us = 1e6;

        std::cerr << std::fixed << std::setprecision(2);
        std::cerr << "\n[phase breakdown — avg per query, rank 0 view]\n";
        std::cerr << "  bcast   (query vector → all procs) : "
                  << tm.t_bcast_s  / N * us << " us\n";
        std::cerr << "  coarse  (local centroid rank)       : "
                  << tm.t_coarse_s / N * us << " us\n";
        std::cerr << "  fine    (local cluster fine-scan)   : "
                  << tm.t_fine_s   / N * us << " us\n";
        std::cerr << "  gather  (MPI_Gather results)        : "
                  << tm.t_gather_s / N * us << " us\n";
        std::cerr << "  merge   (top-k merge at rank 0)     : "
                  << tm.t_merge_s  / N * us << " us\n";

        const double comm_us = (tm.t_bcast_s + tm.t_gather_s) / N * us;
        const double comp_us = (tm.t_coarse_s + tm.t_fine_s + tm.t_merge_s) / N * us;
        std::cerr << "  ── comm total (bcast+gather)        : " << comm_us << " us"
                  << "  (" << 100.0 * comm_us / avg_lat_us << "%)\n";
        std::cerr << "  ── comp total (coarse+fine+merge)   : " << comp_us << " us"
                  << "  (" << 100.0 * comp_us / avg_lat_us << "%)\n";

        std::cerr << "\n[fine-scan load balance across " << size << " processes]\n";
        std::cerr << "  avg fine/proc : " << fine_sum / size / N * us << " us\n";
        std::cerr << "  max fine/proc : " << fine_max        / N * us << " us\n";
        std::cerr << "  min fine/proc : " << fine_min        / N * us << " us\n";
        std::cerr << "  imbalance     : "
                  << (fine_max - fine_min) / N * us << " us"
                  << "  (" << 100.0 * (fine_max - fine_min) / fine_max << "%)\n";

        std::cout << std::fixed << std::setprecision(5);
        std::cout << "average recall: "       << avg_recall  << "\n";
        std::cout << "average latency (us): " << avg_lat_us  << "\n";
    }

    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    MPI_Finalize();
    return 0;
}
