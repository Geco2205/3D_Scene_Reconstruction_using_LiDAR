// transform_one.h
//
// Rigid transform of one scan point, shared by the CUDA kernel and the CPU test.
//
// Inside the ICP loop the source points are moved by the pose estimated in the
// previous iteration. Doing it on the GPU keeps the points in device memory
// across iterations, so they do not travel back to the CPU. This header holds
// the per-point arithmetic, tagged HD so the kernel and the CPU reference
// compile exactly the same operations.
//
// The arithmetic reproduces Sophus::SE3d::operator*(point) from Sophus 1.22.10,
// which is so3() * p + translation(), where SO3d::operator* is:
//     uv = q.vec().cross(p);
//     uv += uv;
//     return p + q.w() * uv + q.vec().cross(uv);
// No rotation matrix is built: R * p rounds differently and would not match
// the CPU bit for bit.
//
// No std::vector and no dynamic allocation: this runs inside a CUDA kernel.

#pragma once

#ifndef HD
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif
#endif

// Pose layout: {qx, qy, qz, qw, tx, ty, tz}. The quaternion is (x, y, z, w),
// the same order as Eigen::Quaternion::coeffs(), and must be unit norm, as
// Sophus guarantees.
#ifndef GPU_POSE_SIZE
#define GPU_POSE_SIZE 7
#endif

// Applies a rigid transform to one point.
//
// Receives: pose, GPU_POSE_SIZE doubles {qx, qy, qz, qw, tx, ty, tz}; in,
//           pointer to 3 doubles, the point; out, pointer to 3 doubles for the
//           result.
// Returns : nothing. Writes the transformed point into out.
//
// The point is first copied into local variables, so out may be the same
// array as in (in-place transform).
//
// The cross products follow Eigen's MatrixBase::cross component by component:
//     c0 = a1*b2 - a2*b1,  c1 = a2*b0 - a0*b2,  c2 = a0*b1 - a1*b0
// and every sum is evaluated in the order of the Sophus expression:
//     out_i = ((p_i + qw*uv_i) + (q x uv)_i) + t_i
HD inline void transform_one(const double* pose, const double* in, double* out) {
    // Copy first: in and out may alias.
    const double px = in[0];
    const double py = in[1];
    const double pz = in[2];

    const double qx = pose[0];
    const double qy = pose[1];
    const double qz = pose[2];
    const double qw = pose[3];

    // uv = q.vec().cross(p)
    double ux = qy * pz - qz * py;
    double uy = qz * px - qx * pz;
    double uz = qx * py - qy * px;

    // uv += uv
    ux += ux;
    uy += uy;
    uz += uz;

    // q.vec().cross(uv)
    const double cx = qy * uz - qz * uy;
    const double cy = qz * ux - qx * uz;
    const double cz = qx * uy - qy * ux;

    // (p + q.w() * uv) + q.vec().cross(uv), then + translation()
    out[0] = ((px + qw * ux) + cx) + pose[4];
    out[1] = ((py + qw * uy) + cy) + pose[5];
    out[2] = ((pz + qw * uz) + cz) + pose[6];
}
