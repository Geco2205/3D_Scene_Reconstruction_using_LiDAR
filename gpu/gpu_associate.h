// gpu_associate.h
//
// Public interface of the GPU data-association module for KISS-ICP.
//
// Replaces the core of DataAssociation: for every scan point it finds the
// nearest neighbour in the voxel map. The rest of the registration stage
// (linear system, solver, pose update) stays on the CPU.
//
// The interface is plain C so it can be linked from code compiled by a normal
// C++ compiler, without nvcc. Buffers are allocated once in gpu_init and reused
// across scans, because the ICP loop calls gpu_run many times per scan and
// allocating on every call would add avoidable overhead.
//
// Typical use:
//
//     gpu_init(max_vox, max_pts, max_src);     // once
//     for (each scan) {
//         gpu_set_map(voxel_size, n_vox, coords, start, count, n_pts, pts);
//         for (each ICP iteration) {
//             gpu_run(n, src, max_dist, out_idx);
//         }
//     }
//     gpu_free();                              // once
//
// Every function returning int uses these codes:
//    0  success
//   -1  the map does not fit the allocated arrays, or is inconsistent
//   -2  too many scan points for the allocated buffer
//   -3  a CUDA call failed
//   -4  gpu_run was called with no map loaded

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Allocates the GPU buffers once.
//
// Receives: max_vox, largest voxel count the map may have; max_pts, largest
//           map point count; max_src, largest scan point count per call.
// Returns : 0 on success, -1 if any capacity is not positive, -3 if a CUDA
//           allocation or a host allocation failed.
//
// Reserves the device arrays, creates the CUDA events used for timing, and
// reserves the host scratch memory used to sort the map. If a session was
// already open it is released first, so calling this twice is safe.
int gpu_init(int max_vox, int max_pts, int max_src);

// Uploads the voxel map to the GPU.
//
// Receives: voxel_size, voxel edge in metres; n_vox, number of voxels; coords,
//           start and count, the three parallel arrays describing the map;
//           n_pts, number of map points; pts, flat array of map points.
// Returns : 0 on success, -1 if the module is not initialised or the map does
//           not fit or is inconsistent, -3 if a CUDA call failed.
//
// Validates the sizes and that every start/count pair addresses a valid range
// inside pts, then checks that the map is sorted lexicographically by
// coordinate. If it is not, it sorts it on the host before uploading, because
// find_voxel uses binary search and on unsorted data would return wrong results
// silently.
//
// pts is not reordered. Only coords, start and count are, and start still
// points at the same positions inside pts.
int gpu_set_map(double voxel_size, int n_vox, const int* coords,
                const int* start, const int* count, int n_pts,
                const double* pts);

// Runs the association for one ICP iteration.
//
// Receives: n, number of scan points; src, 3*n doubles (x, y, z per point),
//           already in the map frame; max_dist, largest accepted correspondence
//           distance; out_idx, caller-owned array of n ints to write into.
// Returns : 0 on success, -2 if n is negative, exceeds max_src, or a pointer is
//           null, -4 if the module is not initialised or no map is loaded, -3
//           if a CUDA call failed.
//
// Uploads src, launches the kernel with 128 threads per block, and downloads
// out_idx. Each output entry is the index of the neighbour inside pts, or -1 if
// that point got no match. With n == 0 it returns 0 without launching anything.
int gpu_run(int n, const double* src, double max_dist, int* out_idx);

// Releases everything reserved by gpu_init.
//
// Receives: nothing.
// Returns : nothing.
//
// Destroys the CUDA events, frees the device and host buffers, and resets the
// module state. Safe to call more than once and safe to call on a module that
// was never initialised.
void gpu_free(void);

// Timings of the LAST call to each function, in milliseconds.
//  - Fields marked (cudaEvent) are measured by the GPU with events: they do not
//    include CPU waits or launch overhead.
//  - Fields marked (chrono) are measured by a CPU clock.
// Each field keeps the value from the last call of the function that produces
// it: gpu_set_map fills the first three and gpu_run fills the other four.
typedef struct {
    double sort_ms;      // gpu_set_map: order check + sort_voxels, CPU only (chrono)
    double h2d_map_ms;   // gpu_set_map: copying the map to the GPU (cudaEvent)
    double set_map_ms;   // gpu_set_map: whole call (chrono)
    double h2d_src_ms;   // gpu_run: copying src to the GPU (cudaEvent)
    double kernel_ms;    // gpu_run: kernel (cudaEvent)
    double d2h_ms;       // gpu_run: copying out_idx back to the CPU (cudaEvent)
    double run_ms;       // gpu_run: whole call (chrono)
} GpuTiming;

// Reads back the timings of the last calls.
//
// Receives: t, pointer to a GpuTiming the caller owns.
// Returns : nothing. The struct is filled in place; if t is NULL nothing
//           happens.
//
// Splitting transfer time from kernel time is what allows reporting whether the
// speed-up comes from the computation or is eaten by moving data across the
// bus.
void gpu_get_timing(GpuTiming* t);

#ifdef __cplusplus
}  // extern "C"
#endif
