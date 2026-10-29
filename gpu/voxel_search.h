// voxel_search.h
//
// Lowest layer of the GPU data-association module for KISS-ICP.
//
// The local map is a list of voxels (1 m cubes) kept sorted by coordinate. To
// find the nearest neighbour of a scan point, the search must first locate the
// voxel the point falls into and its 26 neighbours inside that list. This file
// provides the two primitives that locate a voxel: the ordering rule and the
// binary search built on top of it.
//
// Both functions are tagged HD so the exact same code compiles for the CPU and
// for the GPU. That is deliberate: the accelerated version must select the same
// neighbours as the CPU reference, and a single source avoids two
// implementations drifting apart.
//
// No std::vector and no dynamic allocation: everything here runs inside a CUDA
// kernel.

#pragma once

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// Compares two voxel coordinates in lexicographic order: x first, then y, then
// z. Negative values compare by their normal numeric value (-1 is below 2).
//
// Receives: (ax, ay, az) the first coordinate, (bx, by, bz) the second one.
// Returns : -1 if a comes before b, 1 if it comes after, 0 if they are equal.
//
// This is the single source of truth for the ordering. Both sort_voxels and
// find_voxel call it, so the sorted layout and the binary search can never
// disagree about what "sorted" means.
HD inline int cmp_voxel(int ax, int ay, int az, int bx, int by, int bz) {
    if (ax != bx) return (ax < bx) ? -1 : 1;
    if (ay != by) return (ay < by) ? -1 : 1;
    if (az != bz) return (az < bz) ? -1 : 1;
    return 0;
}

// Locates a voxel inside a flat, sorted coordinate list using binary search.
//
// Receives: (a, b, c) the voxel coordinate to look for; coords, a flat array
//           where coords[3*i], coords[3*i+1], coords[3*i+2] are the x, y, z of
//           voxel i; n, the number of voxels in the list.
// Returns : the index i of the voxel, or -1 if it is not in the list.
//
// Keeps a low and a high marker over the range still worth looking at. On each
// step it takes the middle element and asks whether the target is above or
// below it, moving the matching marker. When low passes high nothing is left to
// inspect and the voxel does not exist.
//
// Requires coords to be sorted by cmp_voxel. On unsorted input the search
// silently returns wrong results, which is why gpu_set_map checks the ordering
// before uploading.
HD inline int find_voxel(int a, int b, int c, const int* coords, int n) {
    int lo = 0;
    int hi = n - 1;

    while (lo <= hi) {
        const int mid = lo + (hi - lo) / 2;
        const int r = cmp_voxel(a, b, c,
                                coords[3 * mid],
                                coords[3 * mid + 1],
                                coords[3 * mid + 2]);
        if (r == 0) return mid;
        if (r > 0) lo = mid + 1;   // the target lies after the middle
        else       hi = mid - 1;   // the target lies before the middle
    }
    return -1;
}
