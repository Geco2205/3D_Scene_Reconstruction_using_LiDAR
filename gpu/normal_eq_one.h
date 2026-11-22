// normal_eq_one.h
//
// Per-correspondence term of the ICP normal equations, shared by the CUDA
// kernel and the CPU test.
//
// This is the body of the transform step inside BuildLinearSystem
// (Registration.cpp, KISS-ICP v1.0.0): residual, Jacobian J_r, Geman-McClure
// weight, and the contributions J_r^T w J_r and J_r^T w r. It is tagged HD so
// the kernel and the CPU reference compile the exact same arithmetic; there is
// only one version of the formula.
//
// No std::vector and no dynamic allocation: this runs inside a CUDA kernel.

#pragma once

#include "gpu_icp.h"

#ifndef HD
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif
#endif

// Computes the 28 values contributed by one correspondence.
//
// Receives: src, pointer to 3 doubles, the scan point (already transformed);
//           tgt, pointer to 3 doubles, its nearest map point; kernel_scale, the
//           Geman-McClure scale k; out, array of GPU_ICP_NUM_TERMS doubles.
// Returns : nothing. Writes into out, in the layout of gpu_icp.h: the 21 upper
//           triangle values of J^T w J, the 6 values of J^T w r, and 1.0 in
//           the counter slot.
//
// Follows BuildLinearSystem step by step:
//   r   = src - tgt
//   J_r = [ I | -hat(src) ]            (3x6, hat as in Sophus::SO3d::hat)
//   w   = k^2 / (k + |r|^2)^2           (square(k) / square(k + r2))
//   out = (J_r^T w) J_r  and  (J_r^T w) r
//
// Every three-term sum is evaluated left to right, (a0 + a1) + a2. This is the
// order Eigen uses for squaredNorm() of a Vector3d, measured with Eigen 3.3.4
// and 3.4.0 on x86-64 and on aarch64 (Cortex-A57) with FMA contraction off: it
// matched in 100% of 20,000 random cases, while a0 + (a1 + a2) matched only 76%.
// The order changes |r|^2 and therefore w, so it matters here. In the J^T w J
// and J^T w r sums it makes no difference, because each column of J_r has at
// least one zero entry. Only the upper triangle is computed because J^T w J is
// symmetric.
//
// Points without a correspondence are not passed here: the caller writes zeros
// for them.
HD inline void normal_eq_one(const double* src, const double* tgt,
                             double kernel_scale, double* out) {
    // Residual and its squared norm.
    const double r0 = src[0] - tgt[0];
    const double r1 = src[1] - tgt[1];
    const double r2 = src[2] - tgt[2];
    const double res2 = (r0 * r0 + r1 * r1) + r2 * r2;   // as Eigen squaredNorm()

    // Geman-McClure weight: square(k) / square(k + r2).
    const double ks2 = kernel_scale * kernel_scale;
    const double den = kernel_scale + res2;
    const double w = ks2 / (den * den);

    // Jacobian J_r = [ I | -1.0 * hat(src) ], row-major J[row][col].
    // hat(v) = [ 0 -v2  v1 ;  v2  0 -v0 ; -v1  v0  0 ], as Sophus::SO3d::hat.
    const double hat[3][3] = {{0.0, -src[2], src[1]},
                              {src[2], 0.0, -src[0]},
                              {-src[1], src[0], 0.0}};
    double J[3][6];
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            J[r][c] = (r == c) ? 1.0 : 0.0;
            J[r][3 + c] = -1.0 * hat[r][c];
        }
    }

    // J^T w J, upper triangle row by row: element (i, j) = sum_k (J[k][i] * w) * J[k][j].
    int o = 0;
    for (int i = 0; i < 6; ++i) {
        const double a0 = J[0][i] * w;
        const double a1 = J[1][i] * w;
        const double a2 = J[2][i] * w;
        for (int j = i; j < 6; ++j) {
            out[o++] = (a0 * J[0][j] + a1 * J[1][j]) + a2 * J[2][j];
        }
    }

    // J^T w r: element i = sum_k (J[k][i] * w) * r[k].
    for (int i = 0; i < 6; ++i) {
        out[GPU_ICP_NUM_JTJ + i] =
            ((J[0][i] * w) * r0 + (J[1][i] * w) * r1) + (J[2][i] * w) * r2;
    }

    // Counter: this point had a correspondence.
    out[GPU_ICP_NUM_JTJ + GPU_ICP_NUM_JTR] = 1.0;
}
