// =============================================================================
// hnsw_partition.h -- Partitioned HNSW index
//
// Architecture:
//   The base dataset is randomly shuffled (fixed seed) and split into
//   num_parts equal-sized parts.  One HierarchicalNSW is built on each part.
//   Labels stored in each partition HNSW are original base indices, so search
//   returns original IDs directly without any extra mapping.
//
//   Unlike IVF+HNSW, there is no coarse scan -- every query searches every
//   partition (split across MPI processes), so recall is not limited by a
//   coarse quantization step.
//
// Build:  hnsw_partition_build()  -- shuffle + split + build num_parts HNSWs.
// Save:   hnsw_partition_save()   -- meta.bin + p{p:04d}.hnsw per partition.
// Load:   hnsw_partition_load()   -- load only the caller's assigned range.
// Free:   hnsw_partition_free()   -- release all heap allocations.
//
// Platform: AArch64.
// =============================================================================
#pragma once

#include <vector>
#include <queue>
#include <utility>
#include <cstdint>
#include <cstddef>
#include <fstream>
#include <string>
#include <algorithm>
#include <random>
#include <cstdio>
#include <sys/stat.h>

#include "hnsw_simd.h"   // InnerProductSpaceNEON, HierarchicalNSW


// =============================================================================
// PartitionedHNSWIndex
// =============================================================================
struct PartitionedHNSWIndex {
    size_t num_parts       = 0;
    size_t base_number     = 0;
    size_t part_sz         = 0;   // base_number / num_parts (last part may be larger)
    size_t vecdim          = 0;
    InnerProductSpaceNEON* space = nullptr;
    std::vector<HierarchicalNSW<float>*> parts;  // parts[p] != nullptr iff loaded
    size_t M               = 16;
    size_t ef_construction = 200;
};


// =============================================================================
// hnsw_partition_build -- build all num_parts partition HNSWs from scratch
//
// Called by rank 0 only.  Other ranks wait at an MPI_Barrier in main.
// seed fixes the random shuffle so the same partitioning is reproducible.
// =============================================================================
void hnsw_partition_build(
    PartitionedHNSWIndex& idx,
    const float* base, size_t n, size_t d,
    size_t num_parts,
    size_t M               = 16,
    size_t ef_construction = 200,
    unsigned seed          = 42)
{
    idx.num_parts       = num_parts;
    idx.base_number     = n;
    idx.part_sz         = n / num_parts;
    idx.vecdim          = d;
    idx.M               = M;
    idx.ef_construction = ef_construction;
    idx.space           = new InnerProductSpaceNEON(d);
    idx.parts.assign(num_parts, nullptr);

    // Random permutation with fixed seed for reproducibility
    std::vector<uint32_t> perm(n);
    for (size_t i = 0; i < n; ++i) perm[i] = static_cast<uint32_t>(i);
    std::mt19937 rng(seed);
    std::shuffle(perm.begin(), perm.end(), rng);

    for (size_t p = 0; p < num_parts; ++p) {
        size_t lo  = p * idx.part_sz;
        size_t hi  = (p + 1 == num_parts) ? n : lo + idx.part_sz;
        size_t psz = hi - lo;

        auto* hnsw = new HierarchicalNSW<float>(idx.space, psz, M, ef_construction);
        for (size_t j = 0; j < psz; ++j) {
            uint32_t orig = perm[lo + j];
            hnsw->addPoint(base + static_cast<size_t>(orig) * d,
                           static_cast<hnswlib::labeltype>(orig));
        }
        idx.parts[p] = hnsw;
    }
}


// =============================================================================
// hnsw_partition_save -- persist metadata + all partition HNSW files
//
// Layout inside dir:
//   meta.bin        -- uint32_t[5]: num_parts, base_number, vecdim, M, ef
//   p0000.hnsw ...  -- one hnswlib binary file per partition
//
// meta.bin is written LAST so a partial save is detectable on load.
// =============================================================================
static bool hnsw_partition_save(const PartitionedHNSWIndex& idx,
                                const std::string& dir)
{
    mkdir(dir.c_str(), 0755);

    for (size_t p = 0; p < idx.num_parts; ++p) {
        if (!idx.parts[p]) continue;
        char name[32];
        snprintf(name, sizeof(name), "/p%04zu.hnsw", p);
        idx.parts[p]->saveIndex(dir + name);
    }

    // Write meta.bin last (acts as completion marker)
    {
        std::string path = dir + "/meta.bin";
        std::ofstream f(path, std::ios::binary);
        if (!f) return false;
        uint32_t vals[5] = {
            static_cast<uint32_t>(idx.num_parts),
            static_cast<uint32_t>(idx.base_number),
            static_cast<uint32_t>(idx.vecdim),
            static_cast<uint32_t>(idx.M),
            static_cast<uint32_t>(idx.ef_construction)
        };
        f.write(reinterpret_cast<const char*>(vals), sizeof(vals));
    }
    return true;
}


// =============================================================================
// hnsw_partition_load -- load partition HNSWs for the range [lo_part, hi_part)
//
// vecdim must be supplied (from loading test_query in main).
// Returns false if meta.bin or any required partition file is missing.
// =============================================================================
static bool hnsw_partition_load(
    PartitionedHNSWIndex& idx,
    const std::string& dir,
    size_t vecdim,
    size_t M, size_t ef_construction,
    int lo_part, int hi_part)
{
    {
        std::string path = dir + "/meta.bin";
        std::ifstream f(path, std::ios::binary);
        if (!f.good()) return false;
        uint32_t vals[5];
        f.read(reinterpret_cast<char*>(vals), sizeof(vals));
        idx.num_parts       = vals[0];
        idx.base_number     = vals[1];
        idx.part_sz         = vals[1] / vals[0];
        idx.vecdim          = vecdim;
        idx.M               = M;
        idx.ef_construction = ef_construction;
    }

    idx.space = new InnerProductSpaceNEON(vecdim);
    idx.parts.assign(idx.num_parts, nullptr);

    for (int p = lo_part; p < hi_part; ++p) {
        char name[32];
        snprintf(name, sizeof(name), "/p%04d.hnsw", p);
        std::string path = dir + name;
        { std::ifstream chk(path); if (!chk.good()) return false; }
        idx.parts[static_cast<size_t>(p)] =
            new HierarchicalNSW<float>(idx.space, path);
    }
    return true;
}


// =============================================================================
// hnsw_partition_free -- release all heap memory owned by the index
// =============================================================================
void hnsw_partition_free(PartitionedHNSWIndex& idx)
{
    for (auto* h : idx.parts) delete h;
    idx.parts.clear();
    delete idx.space;
    idx.space = nullptr;
}
