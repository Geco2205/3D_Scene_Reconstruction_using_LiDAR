// associate_one.h
//
// Core of the GPU data-association module for KISS-ICP.
//
// This is the work that the accelerator actually parallelises: the per-point
// nearest-neighbour search taken from DataAssociation in Registration.cpp. One
// call handles one scan point, and the CUDA kernel runs one call per thread.
// Everything else in the registration stage (linear system, solver, pose
// update) stays on the CPU and is outside the scope of this module.
//
// Tagged HD so the same code compiles for CPU and GPU. The CPU build is what
// the GPU results are validated against, so both paths must come from one
// source.
//
// No std::vector and no dynamic allocation: this runs inside a CUDA kernel.

#pragma once

#include <cfloat>
#include <cmath>

#include "voxel_search.h"

// Finds the nearest map point to a single scan point.
//
// Receives: p, pointer to 3 doubles (x, y, z) of the scan point, already in the
//           map frame; coords, start and count, the sorted voxel map as three
//           parallel arrays; n_vox, number of voxels; pts, flat array of map
//           points where pts[3*k..3*k+2] is point k; voxel_size, the edge of a
//           voxel in metres; max_dist, the largest distance accepted as a
//           correspondence.
// Returns : the index k of the nearest point inside pts, or -1 if no point was
//           found within max_dist.
//
// Computes which voxel p falls into, then visits that voxel and its 26
// neighbours. Each neighbour is located with find_voxel; a result of -1 means
// the voxel is empty and is skipped. For every point in a non-empty voxel the
// distance to p is measured, keeping the running best. At the end the best
// candidate is returned only if it exists and is strictly closer than max_dist.
//
// The distance is computed with a square root and compared directly against
// max_dist, exactly as KISS-ICP does. Squared distances are not used: although
// the square root is monotonic, it is not injective in floating point, so two
// candidates with different squared distances can share the same rounded root.
// That would change how ties are broken. The goal here is to match the CPU
// result, which is verified against a KISS-ICP dump.
HD inline int associate_one(const double* p,
                            const int* coords, const int* start, const int* count,
                            int n_vox,
                            const double* pts,
                            double voxel_size, double max_dist) {
    // floor rather than truncation: truncation rounds toward zero, so -0.4
    // would land in voxel 0 when it belongs in voxel -1.
    const int vx = static_cast<int>(std::floor(p[0] / voxel_size));
    const int vy = static_cast<int>(std::floor(p[1] / voxel_size));
    const int vz = static_cast<int>(std::floor(p[2] / voxel_size));

    double best_d = DBL_MAX;
    int best_k = -1;

    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                const int vi = find_voxel(vx + dx, vy + dy, vz + dz, coords, n_vox);
                if (vi < 0) continue;  // that voxel is empty

                const int k0 = start[vi];
                const int k1 = k0 + count[vi];
                for (int k = k0; k < k1; ++k) {
                    const double ex = pts[3 * k]     - p[0];
                    const double ey = pts[3 * k + 1] - p[1];
                    const double ez = pts[3 * k + 2] - p[2];
                    // Same summation order as squaredNorm() in Eigen 3.4 for 3
                    // elements: x^2 + (y^2 + z^2).
                    const double d = std::sqrt(ex * ex + (ey * ey + ez * ez));
                    if (d < best_d) {
                        best_d = d;
                        best_k = k;
                    }
                }
            }
        }
    }

    return (best_k >= 0 && best_d < max_dist) ? best_k : -1;
}
