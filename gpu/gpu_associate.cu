// gpu_associate.cu
//
// CUDA implementation of the data-association module for KISS-ICP.
//
// Holds the module state (device buffers, host scratch, CUDA events), the
// kernel that calls associate_one once per scan point, and the four public
// entry points declared in gpu_associate.h, plus the timing accessor.
//
// The design keeps every allocation in gpu_init and reuses the buffers for the
// whole run. The ICP loop calls gpu_run several times per scan, so allocating
// inside it would add avoidable overhead to every call.
//
// Timing is split between CUDA events, which measure what happens on the GPU
// and its bus, and std::chrono, which measures what happens on the CPU. That
// separation is what lets the report distinguish kernel time from transfer
// time.

#include "gpu_associate.h"

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

// cmp_voxel, find_voxel, associate_one (host and device) and sort_voxels (host only).
#include "voxel_search.h"
#include "sort_voxels.h"
#include "associate_one.h"

namespace {

constexpr int kThreadsPerBlock = 128;

// --- Module state ------------------------------------------------------
// Device buffers, allocated once in gpu_init.
int*    d_coords = nullptr;   // 3 * max_vox
int*    d_start  = nullptr;   // max_vox
int*    d_count  = nullptr;   // max_vox
double* d_pts    = nullptr;   // 3 * max_pts
double* d_src    = nullptr;   // 3 * max_src
int*    d_out    = nullptr;   // max_src

// Host scratch memory used to sort the map when it arrives unsorted.
int* h_coords_sorted = nullptr;  // 3 * max_vox
int* h_start_sorted  = nullptr;  // max_vox
int* h_count_sorted  = nullptr;  // max_vox
int* h_order         = nullptr;  // max_vox

int    g_max_vox = 0;
int    g_max_pts = 0;
int    g_max_src = 0;
int    g_n_vox   = 0;
double g_voxel_size = 0.0;

bool g_initialized = false;
bool g_map_loaded  = false;

// --- Timing ------------------------------------------------------------
// cudaEvent measures what happens on the GPU and its bus; std::chrono measures
// what happens on the CPU. gpu_get_timing returns the values of the last call.
constexpr int kNumEvents = 4;       // gpu_run uses 4 marks; gpu_set_map uses 2
cudaEvent_t g_ev[kNumEvents];
int g_events_created = 0;
GpuTiming g_timing = GpuTiming();

using Clock = std::chrono::steady_clock;

// Elapsed milliseconds since a CPU timestamp.
//
// Receives: t0, the starting time point.
// Returns : the elapsed wall time in milliseconds as a double.
double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Runs a CUDA call, and on failure prints file, line and the error string and
// makes the enclosing function return -3. Only usable inside functions that
// return int.
#define CUDA_TRY(call)                                                   \
    do {                                                                 \
        cudaError_t _e = (call);                                         \
        if (_e != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                         __FILE__, __LINE__, cudaGetErrorString(_e));    \
            return -3;                                                   \
        }                                                                \
    } while (0)

// Checks whether the voxel list is already sorted.
//
// Receives: coords, the flat coordinate array; n_vox, number of voxels.
// Returns : true if the list is in strict lexicographic order, false otherwise.
//
// Checks STRICT ordering. A tie (cmp == 0) means duplicate voxels, which is an
// upstream error; here it is treated as "unsorted" so it at least goes through
// sort_voxels.
bool map_is_sorted(const int* coords, int n_vox) {
    for (int i = 1; i < n_vox; ++i) {
        const int* a = coords + 3 * (i - 1);
        const int* b = coords + 3 * i;
        if (cmp_voxel(a[0], a[1], a[2], b[0], b[1], b[2]) >= 0) return false;
    }
    return true;
}

// Validates that every voxel addresses a valid range inside pts.
//
// Receives: start and count, the per-voxel arrays; n_vox, number of voxels;
//           n_pts, number of map points.
// Returns : true if every range is non-negative and stays inside pts.
//
// Without this check, a corrupt start or count makes the kernel read outside
// pts. Such reads can return garbage silently or end in an illegal-memory-access
// error, and either way the symptom shows up far from the cause.
bool ranges_are_valid(const int* start, const int* count, int n_vox, int n_pts) {
    for (int i = 0; i < n_vox; ++i) {
        if (start[i] < 0 || count[i] < 0) return false;
        if (start[i] + count[i] > n_pts) return false;
    }
    return true;
}

// --- Kernel ------------------------------------------------------------
// One thread per scan point.
//
// Receives: src, 3*n doubles with the scan points; n, their count; coords,
//           start, count and n_vox, the sorted voxel map; pts, the map points;
//           voxel_size and max_dist, the search parameters; out_idx, device
//           array of n ints to write into.
// Returns : nothing. Thread i writes its result into out_idx[i].
//
// Threads past n exit immediately, which covers the last partially filled
// block. All the actual work is in associate_one, shared with the CPU build.
__global__ void associate_kernel(const double* __restrict__ src, int n,
                                 const int* __restrict__ coords,
                                 const int* __restrict__ start,
                                 const int* __restrict__ count,
                                 int n_vox,
                                 const double* __restrict__ pts,
                                 double voxel_size, double max_dist,
                                 int* __restrict__ out_idx) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out_idx[i] = associate_one(&src[3 * i], coords, start, count, n_vox,
                               pts, voxel_size, max_dist);
}

}  // namespace

// =======================================================================
// Allocates the GPU buffers, the CUDA events and the host scratch memory.
//
// Receives: max_vox, max_pts, max_src, the capacities the session will honour.
// Returns : 0 on success, -1 if a capacity is not positive, -3 if an allocation
//           failed.
//
// If a session was already open it is freed first, so re-initialising is clean.
// On a host allocation failure it releases everything already taken before
// returning, to avoid leaking device memory.
extern "C" int gpu_init(int max_vox, int max_pts, int max_src) {
    if (g_initialized) gpu_free();  // clean re-init

    if (max_vox <= 0 || max_pts <= 0 || max_src <= 0) return -1;

    g_max_vox = max_vox;
    g_max_pts = max_pts;
    g_max_src = max_src;

    CUDA_TRY(cudaMalloc(&d_coords, sizeof(int) * 3 * max_vox));
    CUDA_TRY(cudaMalloc(&d_start,  sizeof(int) * max_vox));
    CUDA_TRY(cudaMalloc(&d_count,  sizeof(int) * max_vox));
    CUDA_TRY(cudaMalloc(&d_pts,    sizeof(double) * 3 * max_pts));
    CUDA_TRY(cudaMalloc(&d_src,    sizeof(double) * 3 * max_src));
    CUDA_TRY(cudaMalloc(&d_out,    sizeof(int) * max_src));

    for (int i = 0; i < kNumEvents; ++i) {
        CUDA_TRY(cudaEventCreate(&g_ev[i]));
        ++g_events_created;
    }

    h_coords_sorted = static_cast<int*>(std::malloc(sizeof(int) * 3 * max_vox));
    h_start_sorted  = static_cast<int*>(std::malloc(sizeof(int) * max_vox));
    h_count_sorted  = static_cast<int*>(std::malloc(sizeof(int) * max_vox));
    h_order         = static_cast<int*>(std::malloc(sizeof(int) * max_vox));

    if (!h_coords_sorted || !h_start_sorted || !h_count_sorted || !h_order) {
        gpu_free();
        return -3;
    }

    g_initialized = true;
    g_map_loaded = false;
    return 0;
}

// =======================================================================
// Validates and uploads the voxel map, sorting it first if necessary.
//
// Receives: voxel_size, n_vox, coords, start, count, n_pts, pts, as described
//           in gpu_associate.h.
// Returns : 0 on success, -1 on a validation failure, -3 if a CUDA call failed.
//
// Records the sorting cost, the map transfer cost and the total call cost into
// the timing struct. The map stays marked as not loaded until every copy has
// succeeded, so a failed upload cannot leave gpu_run running against a half
// written map.
extern "C" int gpu_set_map(double voxel_size, int n_vox, const int* coords,
                           const int* start, const int* count, int n_pts,
                           const double* pts) {
    const auto t_total = Clock::now();
    if (!g_initialized) return -1;
    if (n_vox < 0 || n_pts < 0) return -1;
    if (n_vox > g_max_vox || n_pts > g_max_pts) return -1;
    if (voxel_size <= 0.0) return -1;
    if (n_vox > 0 && (!coords || !start || !count)) return -1;
    if (n_pts > 0 && !pts) return -1;
    if (!ranges_are_valid(start, count, n_vox, n_pts)) return -1;

    const int* up_coords = coords;
    const int* up_start  = start;
    const int* up_count  = count;

    // If it arrives unsorted, sort it on the host. pts is left untouched: only
    // which voxel each entry describes is reordered, not the points themselves.
    const auto t_sort = Clock::now();
    if (n_vox > 1 && !map_is_sorted(coords, n_vox)) {
        sort_voxels(n_vox, coords, start, count,
                    h_coords_sorted, h_start_sorted, h_count_sorted, h_order);
        up_coords = h_coords_sorted;
        up_start  = h_start_sorted;
        up_count  = h_count_sorted;
    }
    g_timing.sort_ms = ms_since(t_sort);

    g_map_loaded = false;  // stays false if any copy below fails

    CUDA_TRY(cudaEventRecord(g_ev[0], 0));
    if (n_vox > 0) {
        CUDA_TRY(cudaMemcpy(d_coords, up_coords, sizeof(int) * 3 * n_vox,
                            cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_start, up_start, sizeof(int) * n_vox,
                            cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_count, up_count, sizeof(int) * n_vox,
                            cudaMemcpyHostToDevice));
    }
    if (n_pts > 0) {
        CUDA_TRY(cudaMemcpy(d_pts, pts, sizeof(double) * 3 * n_pts,
                            cudaMemcpyHostToDevice));
    }
    CUDA_TRY(cudaEventRecord(g_ev[1], 0));
    CUDA_TRY(cudaEventSynchronize(g_ev[1]));
    float ms_map = 0.0f;
    CUDA_TRY(cudaEventElapsedTime(&ms_map, g_ev[0], g_ev[1]));
    g_timing.h2d_map_ms = ms_map;

    g_voxel_size = voxel_size;
    g_n_vox = n_vox;
    g_map_loaded = true;
    g_timing.set_map_ms = ms_since(t_total);
    return 0;
}

// =======================================================================
// Runs one association pass: upload src, launch the kernel, download out_idx.
//
// Receives: n, src, max_dist, out_idx, as described in gpu_associate.h.
// Returns : 0 on success, -2 on a bad argument, -4 if there is no map loaded,
//           -3 if a CUDA call failed.
//
// Four events bracket the three phases so transfer and compute can be reported
// separately. The grid is sized so that every point gets a thread, rounding the
// last block up.
extern "C" int gpu_run(int n, const double* src, double max_dist, int* out_idx) {
    if (n == 0) return 0;          // nothing to do, no kernel launched
    if (n < 0) return -2;
    if (!g_initialized) return -4;
    if (!g_map_loaded) return -4;
    if (n > g_max_src) return -2;
    if (!src || !out_idx) return -2;

    const auto t_total = Clock::now();
    CUDA_TRY(cudaEventRecord(g_ev[0], 0));
    CUDA_TRY(cudaMemcpy(d_src, src, sizeof(double) * 3 * n,
                        cudaMemcpyHostToDevice));
    CUDA_TRY(cudaEventRecord(g_ev[1], 0));

    const int blocks = (n + kThreadsPerBlock - 1) / kThreadsPerBlock;
    associate_kernel<<<blocks, kThreadsPerBlock>>>(
        d_src, n, d_coords, d_start, d_count, g_n_vox,
        d_pts, g_voxel_size, max_dist, d_out);

    // cudaGetLastError catches launch failures (invalid configuration, too many
    // registers). Execution failures surface in the following memcpy, which
    // synchronises.
    CUDA_TRY(cudaGetLastError());
    CUDA_TRY(cudaEventRecord(g_ev[2], 0));

    CUDA_TRY(cudaMemcpy(out_idx, d_out, sizeof(int) * n,
                        cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaEventRecord(g_ev[3], 0));
    CUDA_TRY(cudaEventSynchronize(g_ev[3]));

    float ms_h2d = 0.0f, ms_kernel = 0.0f, ms_d2h = 0.0f;
    CUDA_TRY(cudaEventElapsedTime(&ms_h2d,    g_ev[0], g_ev[1]));
    CUDA_TRY(cudaEventElapsedTime(&ms_kernel, g_ev[1], g_ev[2]));
    CUDA_TRY(cudaEventElapsedTime(&ms_d2h,    g_ev[2], g_ev[3]));
    g_timing.h2d_src_ms = ms_h2d;
    g_timing.kernel_ms  = ms_kernel;
    g_timing.d2h_ms     = ms_d2h;
    g_timing.run_ms     = ms_since(t_total);
    return 0;
}

// =======================================================================
// Releases every resource and resets the module state.
//
// Receives: nothing.
// Returns : nothing.
//
// Destroys only the events that were actually created, so it is safe to call
// after a partially failed gpu_init. Every pointer is nulled and every counter
// zeroed, which also makes a second call harmless.
extern "C" void gpu_free(void) {
    for (int i = 0; i < g_events_created; ++i) cudaEventDestroy(g_ev[i]);
    g_events_created = 0;
    g_timing = GpuTiming();

    cudaFree(d_coords); d_coords = nullptr;
    cudaFree(d_start);  d_start  = nullptr;
    cudaFree(d_count);  d_count  = nullptr;
    cudaFree(d_pts);    d_pts    = nullptr;
    cudaFree(d_src);    d_src    = nullptr;
    cudaFree(d_out);    d_out    = nullptr;

    std::free(h_coords_sorted); h_coords_sorted = nullptr;
    std::free(h_start_sorted);  h_start_sorted  = nullptr;
    std::free(h_count_sorted);  h_count_sorted  = nullptr;
    std::free(h_order);         h_order         = nullptr;

    g_max_vox = g_max_pts = g_max_src = 0;
    g_n_vox = 0;
    g_voxel_size = 0.0;
    g_initialized = false;
    g_map_loaded = false;
}

// =======================================================================
// Copies out the timings of the last calls.
//
// Receives: t, pointer to a caller-owned GpuTiming.
// Returns : nothing. Does nothing if t is NULL.
extern "C" void gpu_get_timing(GpuTiming* t) {
    if (t) *t = g_timing;
}
