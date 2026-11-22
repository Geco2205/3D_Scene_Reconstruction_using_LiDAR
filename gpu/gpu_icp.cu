// gpu_icp.cu
//
// CUDA implementation of the normal-equations module for KISS-ICP
// (Acelerador II).
//
// Holds the module state (per-block partial sums on device and host), the
// block-reduction kernel, and the entry points declared in gpu_icp.h.
//
// The reduction is deterministic on purpose. double atomicAdd does not exist on
// sm_53 (Jetson Nano), and even where it exists its summation order depends on
// thread scheduling. Instead, each block reduces with a fixed shared-memory
// tree and writes its 28 partial sums; the host adds the partials in block
// order. The same input always produces the same bits.

#include "gpu_icp.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "normal_eq_one.h"
#include "transform_one.h"

namespace {

constexpr int kThreadsPerBlock = 128;
constexpr int kTerms = GPU_ICP_NUM_TERMS;

// The tree halves the active range every step, so the block size must be a
// power of two.
static_assert((kThreadsPerBlock & (kThreadsPerBlock - 1)) == 0,
              "kThreadsPerBlock must be a power of two");

// Shared memory: 28 * 128 doubles = 28 KB per block, below the 48 KB static
// limit of sm_53.
static_assert(kTerms * kThreadsPerBlock * sizeof(double) <= 48 * 1024,
              "shared reduction buffer exceeds 48 KB");

// --- Module state ------------------------------------------------------
double* d_partial = nullptr;   // max_blocks * 28, written by the kernel
double* h_partial = nullptr;   // max_blocks * 28, host copy for the final sum

int  g_max_n = 0;
int  g_max_blocks = 0;
bool g_initialized = false;

// Runs a CUDA call, and on failure prints file, line and the error string and
// makes the enclosing function return -3. Only usable inside functions that
// return int. Same macro as gpu_associate.cu.
#define CUDA_TRY(call)                                                   \
    do {                                                                 \
        cudaError_t _e = (call);                                         \
        if (_e != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                         __FILE__, __LINE__, cudaGetErrorString(_e));    \
            return -3;                                                   \
        }                                                                \
    } while (0)

// --- Kernel ------------------------------------------------------------
// One thread per scan point, 128 threads per block.
//
// Receives: src, 3*n doubles with the transformed scan points; pts, the map
//           points; idx, n ints with the neighbour of each point or -1; n, the
//           number of points; kernel_scale, the Geman-McClure scale;
//           partial, device array of gridDim.x * 28 doubles.
// Returns : nothing. Block b writes its 28 sums into partial[28*b .. 28*b+27].
//
// Each thread computes its 28 values with normal_eq_one, or 28 zeros when it is
// past n (last, partially filled block) or its point has no correspondence.
// Those threads still take part in the reduction: an early return would skip
// __syncthreads, which is undefined behaviour, and would leave their shared
// slots holding garbage from a previous launch.
//
// The values go to shared memory as sh[term * 128 + tid], so consecutive
// threads touch consecutive addresses. The tree then adds pairs at distance 64,
// 32, ..., 1, with a barrier after each level; the pairing is fixed, so the
// order of the additions never changes.
__global__ void normal_eq_kernel(const double* __restrict__ src,
                                 const double* __restrict__ pts,
                                 const int* __restrict__ idx,
                                 int n, double kernel_scale,
                                 double* __restrict__ partial) {
    __shared__ double sh[kTerms * kThreadsPerBlock];

    const int tid = threadIdx.x;
    const int i = blockIdx.x * kThreadsPerBlock + tid;

    double v[kTerms];
    for (int t = 0; t < kTerms; ++t) v[t] = 0.0;

    if (i < n) {
        const int j = idx[i];
        if (j >= 0) normal_eq_one(&src[3 * i], &pts[3 * j], kernel_scale, v);
    }

    for (int t = 0; t < kTerms; ++t) sh[t * kThreadsPerBlock + tid] = v[t];
    __syncthreads();

    for (int s = kThreadsPerBlock / 2; s > 0; s >>= 1) {
        if (tid < s) {
            for (int t = 0; t < kTerms; ++t) {
                sh[t * kThreadsPerBlock + tid] += sh[t * kThreadsPerBlock + tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        for (int t = 0; t < kTerms; ++t) {
            partial[blockIdx.x * kTerms + t] = sh[t * kThreadsPerBlock];
        }
    }
}

// Pose wrapper so the 7 doubles travel to the kernel by value, as a kernel
// parameter, with no cudaMemcpy and no device buffer.
static_assert(GPU_POSE_SIZE == 7, "pose layout is {qx, qy, qz, qw, tx, ty, tz}");
struct PoseArg {
    double v[GPU_POSE_SIZE];
};

// One thread per scan point, 128 threads per block.
//
// Receives: src, 3*n doubles, read and overwritten in place; n, the number of
//           points; pose, the transform by value.
// Returns : nothing. Thread i replaces point i with its transformed value.
//
// Threads past n exit right away; there is no __syncthreads in this kernel, so
// an early return is safe here. Each point is read and written only by its own
// thread, and transform_one copies the input to locals before writing, so the
// in-place update is race free.
__global__ void transform_kernel(double* src, int n, PoseArg pose) {
    const int i = blockIdx.x * kThreadsPerBlock + threadIdx.x;
    if (i >= n) return;
    transform_one(pose.v, &src[3 * i], &src[3 * i]);
}

}  // namespace

// =======================================================================
// Allocates the partial-sum buffers on device and host.
//
// Receives: max_n, the capacity in scan points.
// Returns : 0 on success, -1 if max_n is not positive, -3 if an allocation
//           failed.
//
// One block handles 128 points, so ceil(max_n / 128) blocks of 28 doubles are
// enough. On any failure everything already reserved is released.
extern "C" int gpu_icp_init(int max_n) {
    if (g_initialized) gpu_icp_free();  // clean re-init

    if (max_n <= 0) return -1;

    const int max_blocks = (max_n + kThreadsPerBlock - 1) / kThreadsPerBlock;
    const size_t bytes = sizeof(double) * kTerms * static_cast<size_t>(max_blocks);

    cudaError_t e = cudaMalloc(&d_partial, bytes);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",
                     __FILE__, __LINE__, cudaGetErrorString(e));
        gpu_icp_free();
        return -3;
    }

    h_partial = static_cast<double*>(std::malloc(bytes));
    if (!h_partial) {
        gpu_icp_free();
        return -3;
    }

    g_max_n = max_n;
    g_max_blocks = max_blocks;
    g_initialized = true;
    return 0;
}

// =======================================================================
// Releases the buffers and resets the module state.
//
// Receives: nothing.
// Returns : nothing.
//
// Every pointer is nulled, so a second call is harmless.
extern "C" void gpu_icp_free(void) {
    cudaFree(d_partial);  d_partial = nullptr;
    std::free(h_partial); h_partial = nullptr;
    g_max_n = 0;
    g_max_blocks = 0;
    g_initialized = false;
}

// =======================================================================
// Builds the normal equations for one ICP iteration.
//
// Receives: n, d_src, d_pts, d_idx, kernel_scale, out, as described in
//           gpu_icp.h.
// Returns : 0 on success, -1 on an invalid argument, -2 on an invalid n, -3 if a
//           CUDA call failed, -4 if the module is not initialised.
//
// Launches ceil(n / 128) blocks, downloads their partial sums and adds them in
// block order: out[t] = partial[0][t] + partial[1][t] + ... from left to right.
// The download is a blocking cudaMemcpy, so it also waits for the kernel and
// surfaces any execution error. The kernel runs on the default stream, so it
// is ordered after whatever wrote d_src and d_idx on that same stream.
extern "C" int gpu_normal_eq(int n, const double* d_src, const double* d_pts,
                             const int* d_idx, double kernel_scale, double* out) {
    if (!g_initialized) return -4;
    if (!out) return -1;
    if (n < 0 || n > g_max_n) return -2;

    if (n == 0) {  // empty sum: zeros, nothing launched
        for (int t = 0; t < kTerms; ++t) out[t] = 0.0;
        return 0;
    }

    if (!d_src || !d_pts || !d_idx) return -1;
    if (!std::isfinite(kernel_scale)) return -1;

    const int blocks = (n + kThreadsPerBlock - 1) / kThreadsPerBlock;

    normal_eq_kernel<<<blocks, kThreadsPerBlock>>>(d_src, d_pts, d_idx, n,
                                                   kernel_scale, d_partial);
    // Launch failures (invalid configuration, too many resources) show up here;
    // execution failures show up in the blocking memcpy below.
    CUDA_TRY(cudaGetLastError());

    CUDA_TRY(cudaMemcpy(h_partial, d_partial,
                        sizeof(double) * kTerms * static_cast<size_t>(blocks),
                        cudaMemcpyDeviceToHost));

    // Fixed-order final sum over blocks. Starting from block 0 instead of 0.0
    // keeps a single-block result an exact copy of that block.
    for (int t = 0; t < kTerms; ++t) {
        double acc = h_partial[t];
        for (int b = 1; b < blocks; ++b) acc += h_partial[b * kTerms + t];
        out[t] = acc;
    }
    return 0;
}

// =======================================================================
// Transforms the scan points in device memory, in place.
//
// Receives: n, d_src, pose, as described in gpu_icp.h.
// Returns : 0 on success, -1 on a null pointer or a non-finite pose value, -2 on
//           an invalid n, -3 if a CUDA call failed, -4 if the module is not
//           initialised.
//
// Checks follow gpu_normal_eq: the host argument (pose) is checked before the
// n == 0 shortcut, the device pointer only when there is work to launch. The
// pose is copied into a PoseArg and passed by value. The final
// cudaStreamSynchronize makes an execution error surface here, as -3, instead
// of in the next call; ordering alone would not need it, because every kernel
// of the module runs on the default stream.
extern "C" int gpu_transform(int n, double* d_src, const double* pose) {
    if (!g_initialized) return -4;
    if (!pose) return -1;
    if (n < 0 || n > g_max_n) return -2;

    if (n == 0) return 0;  // nothing to transform, nothing launched

    if (!d_src) return -1;

    PoseArg p;
    for (int k = 0; k < GPU_POSE_SIZE; ++k) {
        if (!std::isfinite(pose[k])) return -1;
        p.v[k] = pose[k];
    }

    const int blocks = (n + kThreadsPerBlock - 1) / kThreadsPerBlock;
    transform_kernel<<<blocks, kThreadsPerBlock>>>(d_src, n, p);

    CUDA_TRY(cudaGetLastError());         // launch failures
    CUDA_TRY(cudaStreamSynchronize(0));   // execution failures
    return 0;
}
