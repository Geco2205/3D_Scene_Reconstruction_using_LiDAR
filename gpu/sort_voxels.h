// sort_voxels.h
//
// Host-side helper of the GPU data-association module for KISS-ICP.
//
// find_voxel uses binary search, which only works on a sorted list. The voxel
// map, however, comes out of a hash container and carries no ordering
// guarantee. This file sorts the map so the search can be used.
//
// The map is described by three parallel arrays (coords, start, count) that
// must stay aligned entry by entry. Sorting them independently would break that
// alignment, so the sort is done once over a permutation and then applied to
// the three arrays together.
//
// Host only: it uses std::sort, so it does not compile for the GPU. This is not
// a limitation because the map is sorted once per scan on the CPU, before the
// upload, and not inside the ICP loop.

#pragma once

#include <algorithm>
#include <numeric>

#include "voxel_search.h"

// Reorders the voxel map into lexicographic coordinate order.
//
// Receives: n_vox, number of voxels; coords, start and count, the three
//           parallel input arrays describing voxel i as coords[3*i..3*i+2],
//           start[i] and count[i]; coords_out, start_out and count_out, the
//           output arrays, sized 3*n_vox and n_vox; order, scratch memory of
//           n_vox ints supplied by the caller.
// Returns : nothing. The result is written into the three output arrays.
//
// Works in three steps: build the list of turns 0..n_vox-1, sort those turns by
// comparing the coordinates of each voxel with cmp_voxel, and finally copy the
// three arrays following that order. Indices are sorted rather than the data
// itself, so each swap moves one int instead of five values.
//
// pts is not touched and start is copied unchanged: its values still point at
// the same positions inside pts. What changes is only which voxel owns each
// entry.
//
// The output arrays must not alias the input arrays: the copy step would read
// data it has already overwritten.
inline void sort_voxels(int n_vox,
                        const int* coords, const int* start, const int* count,
                        int* coords_out, int* start_out, int* count_out,
                        int* order) {
    // 1. List of turns: 0, 1, ..., n_vox - 1
    std::iota(order, order + n_vox, 0);

    // 2. Sort the turns by the coordinates of each voxel.
    //    Indices are reordered, not data: moving one int is cheaper than moving
    //    three coordinates plus start plus count.
    std::sort(order, order + n_vox, [coords](int a, int b) {
        return cmp_voxel(coords[3 * a], coords[3 * a + 1], coords[3 * a + 2],
                         coords[3 * b], coords[3 * b + 1], coords[3 * b + 2]) < 0;
    });

    // 3. Build the three new lists following the order of the turns.
    for (int i = 0; i < n_vox; ++i) {
        const int src = order[i];
        coords_out[3 * i]     = coords[3 * src];
        coords_out[3 * i + 1] = coords[3 * src + 1];
        coords_out[3 * i + 2] = coords[3 * src + 2];
        start_out[i] = start[src];
        count_out[i] = count[src];
    }
}
