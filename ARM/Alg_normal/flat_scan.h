// 禁止修改该文件
// =============================================================================
// flat_scan.h — Reference baseline: exact brute-force k-NN search
//
// Purpose:
//   Provides the ground-truth search function `flat_search` used to measure
//   recall of any approximate algorithm you implement in main.cc.
//   This file is READ-ONLY — do NOT modify it.
//
// Distance metric:
//   Inner-Product (IP) distance = 1 - dot(query, base_i).
//   The DEEP100K dataset uses normalized vectors, so IP distance is equivalent
//   to cosine distance and maximizing dot product finds nearest neighbors.
//
// Return value convention:
//   Returns a max-heap (priority_queue) ordered by distance descending.
//   The caller drains the heap with .top()/.pop() to read results from
//   farthest-first to closest-last, or vice versa.
// =============================================================================
#pragma once
#include <queue>


// flat_search — exhaustive linear scan over all base vectors
//
// Parameters:
//   base        : pointer to base dataset, row-major layout [base_number × vecdim]
//   query       : pointer to a single query vector of length vecdim
//   base_number : total number of vectors in the base dataset
//   vecdim      : dimensionality of each vector
//   k           : number of nearest neighbors to return
//
// Algorithm:
//   For each base vector i:
//     1. Compute dot product with query  →  dis = sum(base[i][d] * query[d])
//     2. Convert to distance            →  dis = 1 - dot  (IP distance)
//     3. Maintain a max-heap of size k:
//        - If heap has fewer than k elements, push unconditionally.
//        - Otherwise, replace the top (farthest so far) only if this vector
//          is closer (dis < q.top().first), keeping only the k closest.
//
// Returns: max-heap of (distance, index) pairs for the k nearest neighbors.
std::priority_queue<std::pair<float, uint32_t> > flat_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k)
{
    // Max-heap: pair<distance, vector_index>. Top element is the farthest
    // among the current k candidates, enabling O(log k) replacement.
    std::priority_queue<std::pair<float, uint32_t> > q;

    for(int i = 0; i < base_number; ++i)
    {
        float dis = 0;

        // DEEP100K数据集使用ip距离
        // Scalar dot product over vecdim dimensions.
        // base layout: row i starts at base + i*vecdim.
        for(int d = 0; d < vecdim; ++d)
        {
            dis += base[d + i*vecdim]*query[d];
        }
        // Convert similarity to distance: smaller value = closer neighbor.
        dis = 1 - dis;

        if(q.size() < k)
        {
            // Heap not full yet — always accept.
            q.push({dis, i});
        }
        else
        {
            // Only replace the current farthest if this vector is closer.
            if(dis < q.top().first)
            {
                q.push({dis, i});
                q.pop();  // Remove the now-excess farthest element.
            }
        }
    }
    return q;
}
