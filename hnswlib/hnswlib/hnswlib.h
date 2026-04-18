// =============================================================================
// hnswlib.h — Core interfaces, SIMD capability detection, and shared utilities
//
// Purpose:
//   This is the top-level header that every other hnswlib file includes.
//   It defines:
//     1. Compile-time SIMD feature macros (USE_SSE, USE_AVX, USE_AVX512)
//        and runtime CPU/OS capability checks (AVXCapable, AVX512Capable).
//     2. Abstract base classes that decouple the algorithm from specific
//        distance functions and search behaviors:
//          - SpaceInterface<MTYPE>          : distance function provider
//          - AlgorithmInterface<dist_t>     : index add/search interface
//          - BaseFilterFunctor              : per-result filter predicate
//          - BaseSearchStopCondition<dist_t>: pluggable search termination
//     3. Shared binary I/O helpers (writeBinaryPOD / readBinaryPOD).
//     4. pairGreater<T>: min-heap comparator for priority queues.
//
//   After defining these primitives, it includes the concrete implementations:
//     space_l2.h, space_ip.h, stop_condition.h, bruteforce.h, hnswalg.h
// =============================================================================
#pragma once

// https://github.com/nmslib/hnswlib/pull/508
// This allows others to provide their own error stream (e.g. RcppHNSW)
#ifndef HNSWLIB_ERR_OVERRIDE
  #define HNSWERR std::cerr
#else
  #define HNSWERR HNSWLIB_ERR_OVERRIDE
#endif

// SIMD feature macros — set at compile time based on compiler-defined macros.
// Define NO_MANUAL_VECTORIZATION to disable all SIMD paths (scalar fallback).
// USE_SSE  → 128-bit XMM registers (processes 4 floats at once)
// USE_AVX  → 256-bit YMM registers (processes 8 floats at once)
// USE_AVX512 → 512-bit ZMM registers (processes 16 floats at once)
#ifndef NO_MANUAL_VECTORIZATION
#if (defined(__SSE__) || _M_IX86_FP > 0 || defined(_M_AMD64) || defined(_M_X64))
#define USE_SSE
#ifdef __AVX__
#define USE_AVX
#ifdef __AVX512F__
#define USE_AVX512
#endif
#endif
#endif
#endif

#if defined(USE_AVX) || defined(USE_SSE)
#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>
static void cpuid(int32_t out[4], int32_t eax, int32_t ecx) {
    __cpuidex(out, eax, ecx);
}
static __int64 xgetbv(unsigned int x) {
    return _xgetbv(x);
}
#else
#include <x86intrin.h>
#include <cpuid.h>
#include <stdint.h>
static void cpuid(int32_t cpuInfo[4], int32_t eax, int32_t ecx) {
    __cpuid_count(eax, ecx, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
}
static uint64_t xgetbv(unsigned int index) {
    uint32_t eax, edx;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
    return ((uint64_t)edx << 32) | eax;
}
#endif

#if defined(USE_AVX512)
#include <immintrin.h>
#endif

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#define PORTABLE_ALIGN64 __declspec(align(64))
#endif

// Adapted from https://github.com/Mysticial/FeatureDetector
#define _XCR_XFEATURE_ENABLED_MASK  0

// AVXCapable — runtime check: does this CPU+OS combo support AVX?
//
// Two conditions must BOTH be true:
//   1. CPU reports AVX support via CPUID leaf 1, ECX bit 28.
//   2. OS has enabled XSAVE/XRSTORE (bit 27) and the XCR0 register has
//      bits 1 and 2 set (YMM state saved), confirming the OS saves AVX state
//      on context switches. Without OS support, using YMM registers would
//      silently corrupt state in other threads.
static bool AVXCapable() {
    int cpuInfo[4];

    // CPU support
    cpuid(cpuInfo, 0, 0);
    int nIds = cpuInfo[0];

    bool HW_AVX = false;
    if (nIds >= 0x00000001) {
        cpuid(cpuInfo, 0x00000001, 0);
        HW_AVX = (cpuInfo[2] & ((int)1 << 28)) != 0;
    }

    // OS support
    cpuid(cpuInfo, 1, 0);

    bool osUsesXSAVE_XRSTORE = (cpuInfo[2] & (1 << 27)) != 0;
    bool cpuAVXSuport = (cpuInfo[2] & (1 << 28)) != 0;

    bool avxSupported = false;
    if (osUsesXSAVE_XRSTORE && cpuAVXSuport) {
        // XCR0 bits 1:2 = XMM/YMM state must be OS-managed.
        uint64_t xcrFeatureMask = xgetbv(_XCR_XFEATURE_ENABLED_MASK);
        avxSupported = (xcrFeatureMask & 0x6) == 0x6;
    }
    return HW_AVX && avxSupported;
}

// AVX512Capable — runtime check: does this CPU+OS combo support AVX-512?
//
// Requires AVXCapable() as a prerequisite (AVX-512 is a superset of AVX).
// Additionally checks CPUID leaf 7, EBX bit 16 for AVX512F (Foundation).
// XCR0 bits must include opmask (bit 5), ZMM_Hi256 (bit 6), Hi16_ZMM (bit 7)
// — mask 0xe6 checks bits 1,2,5,6,7.
static bool AVX512Capable() {
    if (!AVXCapable()) return false;

    int cpuInfo[4];

    // CPU support
    cpuid(cpuInfo, 0, 0);
    int nIds = cpuInfo[0];

    bool HW_AVX512F = false;
    if (nIds >= 0x00000007) {  //  AVX512 Foundation
        cpuid(cpuInfo, 0x00000007, 0);
        HW_AVX512F = (cpuInfo[1] & ((int)1 << 16)) != 0;
    }

    // OS support
    cpuid(cpuInfo, 1, 0);

    bool osUsesXSAVE_XRSTORE = (cpuInfo[2] & (1 << 27)) != 0;
    bool cpuAVXSuport = (cpuInfo[2] & (1 << 28)) != 0;

    bool avx512Supported = false;
    if (osUsesXSAVE_XRSTORE && cpuAVXSuport) {
        uint64_t xcrFeatureMask = xgetbv(_XCR_XFEATURE_ENABLED_MASK);
        avx512Supported = (xcrFeatureMask & 0xe6) == 0xe6;
    }
    return HW_AVX512F && avx512Supported;
}
#endif

#include <queue>
#include <vector>
#include <iostream>
#include <string.h>

namespace hnswlib {
// labeltype — external user-facing ID type (size_t, typically 64-bit).
// Internally the algorithm uses tableint (uint32_t) to save memory.
typedef size_t labeltype;

// BaseFilterFunctor — optional per-candidate filter applied during search.
// Override operator() to skip unwanted IDs (e.g., ACL filtering, deleted IDs).
// Default implementation accepts everything.
class BaseFilterFunctor {
 public:
    virtual bool operator()(hnswlib::labeltype id) { return true; }
    virtual ~BaseFilterFunctor() {};
};

// BaseSearchStopCondition — pluggable search termination policy.
// Implemented by MultiVectorSearchStopCondition and EpsilonSearchStopCondition
// in stop_condition.h. Used by searchBaseLayerST to support non-standard
// termination criteria beyond simple k-NN.
template<typename dist_t>
class BaseSearchStopCondition {
 public:
    // Called when a new point is added to the result set.
    virtual void add_point_to_result(labeltype label, const void *datapoint, dist_t dist) = 0;
    // Called when a point is removed from the result set (overflow).
    virtual void remove_point_from_result(labeltype label, const void *datapoint, dist_t dist) = 0;
    // Returns true when the search beam can safely terminate.
    virtual bool should_stop_search(dist_t candidate_dist, dist_t lowerBound) = 0;
    // Returns true when a candidate is worth computing distance for.
    virtual bool should_consider_candidate(dist_t candidate_dist, dist_t lowerBound) = 0;
    // Returns true when the result set is over-full and needs trimming.
    virtual bool should_remove_extra() = 0;
    // Post-processing: prune candidates vector to final result set.
    virtual void filter_results(std::vector<std::pair<dist_t, labeltype >> &candidates) = 0;

    virtual ~BaseSearchStopCondition() {}
};

// pairGreater<T> — comparator that makes std::priority_queue a MIN-heap.
// Default priority_queue is a max-heap; applying pairGreater flips the order
// so the top element is the SMALLEST (closest) distance.
template <typename T>
class pairGreater {
 public:
    bool operator()(const T& p1, const T& p2) {
        return p1.first > p2.first;
    }
};

// writeBinaryPOD / readBinaryPOD — type-safe raw binary I/O for POD types.
// Used by saveIndex / loadIndex to serialize index metadata fields.
template<typename T>
static void writeBinaryPOD(std::ostream &out, const T &podRef) {
    out.write((char *) &podRef, sizeof(T));
}

template<typename T>
static void readBinaryPOD(std::istream &in, T &podRef) {
    in.read((char *) &podRef, sizeof(T));
}

// DISTFUNC<MTYPE> — function pointer type for distance functions.
// Signature: (vec1, vec2, param) → distance, where param is typically a
// pointer to the dimension size_t (allows compile-time-unknown dimensions).
template<typename MTYPE>
using DISTFUNC = MTYPE(*)(const void *, const void *, const void *);

// SpaceInterface<MTYPE> — abstract metric space.
// Concrete classes (L2Space, InnerProductSpace) select the best SIMD
// distance function at construction time based on runtime CPU capability.
template<typename MTYPE>
class SpaceInterface {
 public:
    // Returns bytes per vector (dim * sizeof(element)).
    virtual size_t get_data_size() = 0;
    // Returns the chosen distance function pointer (scalar or SIMD variant).
    virtual DISTFUNC<MTYPE> get_dist_func() = 0;
    // Returns a pointer to the dimension parameter passed to the dist func.
    virtual void *get_dist_func_param() = 0;

    virtual ~SpaceInterface() {}
};

// AlgorithmInterface<dist_t> — abstract index interface.
// Both BruteforceSearch and HierarchicalNSW implement this, allowing
// code to swap algorithms without changing the caller.
template<typename dist_t>
class AlgorithmInterface {
 public:
    virtual void addPoint(const void *datapoint, labeltype label, bool replace_deleted = false) = 0;

    // searchKnn — returns k nearest neighbors as a max-heap (farthest-first).
    virtual std::priority_queue<std::pair<dist_t, labeltype>>
        searchKnn(const void*, size_t, BaseFilterFunctor* isIdAllowed = nullptr) const = 0;

    // searchKnnCloserFirst — convenience wrapper: reverses the heap order
    // so results are returned closest-first (index 0 = nearest neighbor).
    virtual std::vector<std::pair<dist_t, labeltype>>
        searchKnnCloserFirst(const void* query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const;

    virtual void saveIndex(const std::string &location) = 0;
    virtual ~AlgorithmInterface(){
    }
};

// searchKnnCloserFirst — default implementation shared by all AlgorithmInterface subclasses.
// Calls searchKnn (which returns farthest-first) then reverses the order by
// filling the result vector backwards from the heap.
template<typename dist_t>
std::vector<std::pair<dist_t, labeltype>>
AlgorithmInterface<dist_t>::searchKnnCloserFirst(const void* query_data, size_t k,
                                                 BaseFilterFunctor* isIdAllowed) const {
    std::vector<std::pair<dist_t, labeltype>> result;

    // here searchKnn returns the result in the order of further first
    auto ret = searchKnn(query_data, k, isIdAllowed);
    {
        size_t sz = ret.size();
        result.resize(sz);
        // Drain the max-heap filling result[] from back to front → closest at [0].
        while (!ret.empty()) {
            result[--sz] = ret.top();
            ret.pop();
        }
    }

    return result;
}
}  // namespace hnswlib

#include "space_l2.h"
#include "space_ip.h"
#include "stop_condition.h"
#include "bruteforce.h"
#include "hnswalg.h"
