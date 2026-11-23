// gpu_icp.h
//
// Public interface of the GPU normal-equations module for KISS-ICP (Acelerador II).
//
// Moves the work of BuildLinearSystem (Registration.cpp) to the GPU: for every
// scan point with a correspondence it builds the Jacobian and the Geman-McClure
// weight, and accumulates J^T w J and J^T w r over all points. Solving the 6x6
// system, SE3::exp, the convergence test, transforming the points and the TSDF
// integration stay on the CPU.
//
// The interface is plain C, like gpu_associate.h, so it links from code built
// by the project compiler without nvcc. Buffers are allocated once in
// gpu_icp_init and reused by every gpu_normal_eq call.
//
// Layout of the 28 output values (GPU_ICP_NUM_TERMS):
//   [ 0..20]  J^T w J, upper triangle row by row:
//             (0,0) (0,1) (0,2) (0,3) (0,4) (0,5)   ->  0..5
//                   (1,1) (1,2) (1,3) (1,4) (1,5)   ->  6..10
//                         (2,2) (2,3) (2,4) (2,5)   -> 11..14
//                               (3,3) (3,4) (3,5)   -> 15..17
//                                     (4,4) (4,5)   -> 18..19
//                                           (5,5)   -> 20
//   [21..26]  J^T w r
//   [27]      number of points with a correspondence (idx >= 0)
//
// Return codes, the same as gpu_associate:
//    0  success
//   -1  invalid argument
//   -2  n is negative or larger than the allocated buffer
//   -3  a CUDA call failed
//   -4  module not initialised

#pragma once

#define GPU_ICP_NUM_TERMS 28   // 21 (J^T w J) + 6 (J^T w r) + 1 (counter)
#define GPU_ICP_NUM_JTJ   21
#define GPU_ICP_NUM_JTR    6

// Pose passed to gpu_transform: {qx, qy, qz, qw, tx, ty, tz}, unit quaternion.
// Also defined in transform_one.h; both definitions are identical.
#ifndef GPU_POSE_SIZE
#define GPU_POSE_SIZE 7
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Allocates the device and host buffers once.
//
// Receives: max_n, largest number of scan points a gpu_normal_eq call will get.
// Returns : 0 on success, -1 if max_n is not positive, -3 if an allocation
//           failed.
//
// Reserves one block of 28 partial sums per 128 scan points, on the device and
// on the host. If a session was already open it is released first.
int gpu_icp_init(int max_n);

// Releases everything reserved by gpu_icp_init.
//
// Receives: nothing.
// Returns : nothing.
//
// Safe to call more than once and on a module that was never initialised.
void gpu_icp_free(void);

// Builds the ICP normal equations on the GPU for one iteration.
//
// Receives: n, number of scan points; d_src, DEVICE array of 3*n doubles with
//           the scan points (already transformed by the current estimate);
//           d_pts, DEVICE array of map points (3 doubles each); d_idx, DEVICE
//           array of n ints, the neighbour of each scan point inside the
//           original pts array or -1 if it has none (what gpu_run writes);
//           kernel_scale, the Geman-McClure scale k; out, HOST array of
//           GPU_ICP_NUM_TERMS doubles to write the totals into.
// Returns : 0 on success, -1 on a null pointer or a non-finite kernel_scale,
//           -2 if n < 0 or n > max_n, -3 if a CUDA call failed, -4 if the
//           module is not initialised.
//
// Launches one thread per point in blocks of 128, reduces each block with a
// fixed tree, downloads the per-block partial sums and adds them on the CPU in
// block order. The result is therefore deterministic: the same input always
// gives the same bits. With n == 0 it writes zeros to out and launches nothing.
//
// Precondition: every idx >= 0 must be a valid position inside d_pts. It is not
// checked on the device; gpu_run only produces valid indices.
int gpu_normal_eq(int n, const double* d_src, const double* d_pts,
                  const int* d_idx, double kernel_scale, double* out);

// Applies a rigid transform, in place, to the scan points in device memory.
//
// Receives: n, number of points; d_src, DEVICE array of 3*n doubles, read and
//           overwritten with the transformed points; pose, HOST array of
//           GPU_POSE_SIZE doubles {qx, qy, qz, qw, tx, ty, tz}, the pose that
//           came out of solving the 6x6 system, with a unit quaternion.
// Returns : 0 on success, -1 on a null pointer or a pose with NaN or infinity,
//           -2 if n < 0 or n > max_n, -3 if a CUDA call failed, -4 if the
//           module is not initialised.
//
// Launches one thread per point in blocks of 128, with the pose passed by
// value to the kernel. Each point gets exactly the arithmetic of
// Sophus::SE3d::operator*(point) (see transform_one.h). With n == 0 it returns
// 0 and launches nothing. The call returns once the kernel has finished, so
// d_src is ready for the next gpu_run or gpu_normal_eq.
//
// The quaternion norm is not checked: like Sophus, it is assumed to be unit.
int gpu_transform(int n, double* d_src, const double* pose);

#ifdef __cplusplus
}  // extern "C"
#endif
