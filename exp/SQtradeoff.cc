// =============================================================================
// SQtradeoff.cc — SQ-SIMD recall/latency tradeoff sweep driver
//
// Sweeps two knobs:
//   argv[1]  p     : coarse-scan candidate count (top-p → rerank → top-k)
//   argv[2]  bits  : quantization bit-width, one of {4, 8, 16}
//
// Example:
//   g++ exp/SQtradeoff.cc -o exp/SQtradeoff -O2 -fopenmp -lpthread -std=c++11
//   ./exp/SQtradeoff 200 8
//
// Output (one line):
//   bits=<b> p=<p> recall=<x.xxxx> avg_latency_us=<y.y> index_MB=<z.zz>
// followed by a CSV line for easy aggregation:
//   CSV,<bits>,<p>,<recall>,<latency_us>,<index_MB>
// =============================================================================

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <iomanip>
#include <sstream>
#include <cstdlib>
#include <sys/time.h>

#include "sq_multibit_simd.h"   // SQIndex4 / SQIndex / SQIndex16 + SIMD search funcs

// Mirrors LoadData<T> from main.cc
template<typename T>
static T* LoadData(const std::string& path, size_t& n, size_t& d)
{
    std::ifstream fin(path, std::ios::in | std::ios::binary);
    fin.read((char*)&n, 4);
    fin.read((char*)&d, 4);
    T* data = new T[n * d];
    int sz = sizeof(T);
    for (size_t i = 0; i < n; ++i) {
        fin.read(((char*)data + i * d * sz), d * sz);
    }
    fin.close();
    std::cerr << "load data " << path
              << "  n=" << n << "  d=" << d
              << "  elem=" << sizeof(T) << "\n";
    return data;
}

struct SearchResult { float recall; int64_t latency_us; };

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <p> <bits>\n"
                  << "  p    : coarse-scan candidate count (p >= k)\n"
                  << "  bits : 4, 8, or 16\n";
        return 1;
    }
    const size_t p    = static_cast<size_t>(std::atoi(argv[1]));
    const int    bits = std::atoi(argv[2]);
    if (bits != 4 && bits != 8 && bits != 16) {
        std::cerr << "bits must be 4, 8, or 16\n";
        return 1;
    }

    // ---- Load DEEP100K ------------------------------------------------------
    size_t test_number = 0, base_number = 0, test_gt_d = 0, vecdim = 0;
    const std::string data_path = "/anndata/";
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin",
                                      test_number, vecdim);
    auto test_gt    = LoadData<int  >(data_path + "DEEP100K.gt.query.100k.top100.bin",
                                      test_number, test_gt_d);
    auto base       = LoadData<float>(data_path + "DEEP100K.base.100k.fbin",
                                      base_number, vecdim);

    test_number = 2000;
    const size_t k = 10;

    // ---- Build the selected SQ index (OUTSIDE the timed loop) ---------------
    SQIndex4  idx4;
    SQIndex   idx8;
    SQIndex16 idx16;
    size_t index_bytes = 0;

    if      (bits == 4)  { idx4 .build(base, base_number, vecdim); index_bytes = idx4 .index_bytes(); }
    else if (bits == 8)  { idx8 .build(base, base_number, vecdim); index_bytes = idx8.codes.size(); }
    else                 { idx16.build(base, base_number, vecdim); index_bytes = idx16.index_bytes(); }

    // ---- Timed query loop ---------------------------------------------------
    std::vector<SearchResult> results(test_number);

    for (size_t i = 0; i < test_number; ++i) {
        const unsigned long USEC = 1000 * 1000;
        struct timeval t0, t1;
        gettimeofday(&t0, nullptr);

        std::priority_queue<std::pair<float, uint32_t>> res;
        const float* q = test_query + i * vecdim;
        if      (bits == 4)  res = sq_search_4bit_simd (idx4,  base, q, k, p);
        else if (bits == 8)  res = sq_flat_search_simd (idx8,  base, q, k, p);
        else                 res = sq_search_16bit_simd(idx16, base, q, k, p);

        gettimeofday(&t1, nullptr);
        int64_t diff = (t1.tv_sec * USEC + t1.tv_usec) -
                       (t0.tv_sec * USEC + t0.tv_usec);

        std::set<uint32_t> gtset;
        for (size_t j = 0; j < k; ++j) {
            gtset.insert(static_cast<uint32_t>(test_gt[j + i * test_gt_d]));
        }
        size_t acc = 0;
        while (!res.empty()) {
            if (gtset.count(res.top().second)) ++acc;
            res.pop();
        }
        results[i] = { static_cast<float>(acc) / static_cast<float>(k), diff };
    }

    double avg_recall = 0.0, avg_latency = 0.0;
    for (size_t i = 0; i < test_number; ++i) {
        avg_recall  += results[i].recall;
        avg_latency += static_cast<double>(results[i].latency_us);
    }
    avg_recall  /= static_cast<double>(test_number);
    avg_latency /= static_cast<double>(test_number);

    const double index_MB = static_cast<double>(index_bytes) / (1024.0 * 1024.0);

    std::cout << std::fixed << std::setprecision(4)
              << "bits=" << bits << " p=" << p
              << " recall=" << avg_recall
              << " avg_latency_us=" << std::setprecision(2) << avg_latency
              << " index_MB="       << std::setprecision(2) << index_MB << "\n";

    // Machine-readable CSV line
    std::cout << "CSV," << bits << "," << p << ","
              << std::setprecision(4) << avg_recall << ","
              << std::setprecision(2) << avg_latency << ","
              << index_MB << "\n";

    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    return 0;
}
