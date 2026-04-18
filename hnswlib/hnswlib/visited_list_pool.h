// =============================================================================
// visited_list_pool.h — Thread-safe pool of "visited" marker arrays
//
// Purpose:
//   During HNSW graph traversal each worker thread needs an array of flags
//   to mark which nodes have already been visited (to avoid re-visiting).
//   A naïve approach would memset the array to 0 before every search, but
//   that is O(N) and dominates cost for large graphs.
//
// Key insight — "generation counter" trick:
//   Instead of clearing the array, VisitedList keeps a counter `curV`.
//   A node is considered "visited" iff mass[node_id] == curV.
//   Resetting just increments curV (O(1)); when it wraps around to 0 a full
//   memset is done once every 65535 resets.
//
// Thread safety:
//   VisitedListPool wraps a deque of pre-allocated VisitedList objects.
//   Threads borrow a list (getFreeVisitedList), use it, and return it
//   (releaseVisitedList). A mutex guards the deque but not the list contents,
//   so concurrent searches each hold their own private list.
// =============================================================================
#pragma once

#include <mutex>
#include <string.h>
#include <deque>

namespace hnswlib {
// vl_type — unsigned short (16-bit) used as the "visited" stamp.
// Using uint16 (max 65535) keeps the array half the size of uint32.
typedef unsigned short int vl_type;

// VisitedList — single visited-node marker array for one search thread.
class VisitedList {
 public:
    vl_type curV;       // current generation stamp; a node is "visited" iff mass[id]==curV
    vl_type *mass;      // marker array, one entry per node in the graph
    unsigned int numelements;

    VisitedList(int numelements1) {
        curV = -1;  // starts at 0xFFFF; first reset() will increment to 0 and memset
        numelements = numelements1;
        mass = new vl_type[numelements];
    }

    // reset — O(1) "clear" using generation counter.
    // Increments curV; the previous stamp value is now stale everywhere.
    // Only memsets when counter wraps to 0 (every 65535 resets).
    void reset() {
        curV++;
        if (curV == 0) {
            memset(mass, 0, sizeof(vl_type) * numelements);
            curV++;  // skip 0 so that uninitialized entries don't appear visited
        }
    }

    ~VisitedList() { delete[] mass; }
};
///////////////////////////////////////////////////////////
//
// Class for multi-threaded pool-management of VisitedLists
//
/////////////////////////////////////////////////////////

// VisitedListPool — thread-safe object pool for VisitedList instances.
// Prevents repeated allocation/deallocation per query; instead reuses
// pre-allocated VisitedList objects across searches.
class VisitedListPool {
    std::deque<VisitedList *> pool;
    std::mutex poolguard;
    int numelements;

 public:
    // Constructor — pre-allocates initmaxpools VisitedList objects.
    // initmaxpools should equal the maximum number of concurrent searches.
    VisitedListPool(int initmaxpools, int numelements1) {
        numelements = numelements1;
        for (int i = 0; i < initmaxpools; i++)
            pool.push_front(new VisitedList(numelements));
    }

    // getFreeVisitedList — borrow a VisitedList for one search.
    // Returns from pool if available; otherwise allocates a new one.
    // Always calls reset() so the caller gets a clean generation stamp.
    VisitedList *getFreeVisitedList() {
        VisitedList *rez;
        {
            std::unique_lock <std::mutex> lock(poolguard);
            if (pool.size() > 0) {
                rez = pool.front();
                pool.pop_front();
            } else {
                // Pool exhausted (more threads than initmaxpools) — allocate on demand.
                rez = new VisitedList(numelements);
            }
        }
        rez->reset();
        return rez;
    }

    // releaseVisitedList — return a borrowed VisitedList back to the pool.
    void releaseVisitedList(VisitedList *vl) {
        std::unique_lock <std::mutex> lock(poolguard);
        pool.push_front(vl);
    }

    ~VisitedListPool() {
        while (pool.size()) {
            VisitedList *rez = pool.front();
            pool.pop_front();
            delete rez;
        }
    }
};
}  // namespace hnswlib
