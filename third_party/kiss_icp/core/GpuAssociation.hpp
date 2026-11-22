// GpuAssociation.hpp - optional GPU offload of DataAssociation (library gpu_assoc).
//
// Compiled only when KISS_ICP_USE_GPU is defined. At runtime it is enabled with the
// environment variable KISS_USE_GPU=1; otherwise Enabled() is false and nothing changes.
//
// The local map does not change during the ICP loop, so it is flattened and uploaded once
// per scan (SetMap), and every iteration only sends the source points (Associate).
// If any GPU call fails, the error is reported on stderr, counted in "gpu_fallbacks", and
// that scan finishes on the CPU, so a GPU problem never silently changes the result.
//
// [NEW] Full ICP loop on the GPU (Acelerador II), enabled with KISS_GPU_LOOP=1 on top of
// KISS_USE_GPU=1. The source points are uploaded once per scan (SetSource) and stay in
// device memory; each iteration runs association + normal equations on the GPU
// (LoopIteration) and moves the points in place (Transform). Only the 28 sums of the
// linear system come back to the CPU, which still solves the 6x6, computes SE3::exp and
// checks convergence. Every method returns false on failure so the caller can fall back
// to the CPU.
//
// [NEW] Verification mode, enabled with KISS_GPU_CHECK=1 on top of KISS_GPU_LOOP=1. It
// keeps a host copy of the source points, moves it with Sophus exactly like
// TransformPoints, and compares each GPU linear system against the CPU one supplied by
// the caller (DataAssociation + BuildLinearSystem). Results go to the ProfileTimers
// counters. It roughly doubles the cost of the loop, so it is meant for validation runs.
#pragma once

#ifdef KISS_ICP_USE_GPU

#include <Eigen/Core>
#include <algorithm>  // [NEW] std::max in the verification mode
#include <cfloat>     // [NEW] DBL_MIN as a guard against division by zero
#include <cmath>      // [NEW] std::llround, std::fabs
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sophus/se3.hpp>  // [NEW] Transform takes a Sophus::SE3d
#include <tuple>           // [NEW] the CPU reference returns a tuple
#include <utility>
#include <vector>

#include "ProfileTimers.hpp"
#include "VoxelHashMap.hpp"
#include "gpu_associate.h"
#include "gpu_icp.h"  // [NEW] gpu_icp_init, gpu_normal_eq, gpu_transform

namespace kiss_icp::gpu {

using Correspondence = std::pair<Eigen::Vector3d, Eigen::Vector3d>;
using Correspondences = std::vector<Correspondence>;

// [NEW] Same types as Eigen::Matrix6d / Eigen::Vector6d in Registration.cpp. They are
// declared here because Registration.cpp defines those aliases after including this file.
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;

// Capacities reserved once on the GPU (about 27 MB in total). They are far above the maxima
// measured on a 300-scan sample (8139 voxels, 131194 map points, 2096 scan points).
// A scan that exceeds them makes the GPU call fail and that scan runs on the CPU.
constexpr int kMaxVoxels = 65536;
constexpr int kMaxMapPoints = 1048576;
constexpr int kMaxSourcePoints = 16384;

class Session {
public:
    static Session &Get() {
        static Session session;
        return session;
    }

    bool enabled() const { return ready_; }

    // [NEW] True when KISS_GPU_LOOP=1 and gpu_icp_init succeeded.
    bool loop_enabled() const { return loop_ready_; }

    // [NEW] True when KISS_GPU_CHECK=1 and the loop is enabled.
    bool check_enabled() const { return check_; }

    // Flattens the voxel map and uploads it. Returns false if the scan must use the CPU.
    bool SetMap(const VoxelHashMap &map) {
        map_loaded_ = false;
        source_loaded_ = false;  // [NEW] a new map invalidates the source of the last scan
        {
            KISS_TIMER("gpu_flatten_map");
            coords_.clear();
            start_.clear();
            count_.clear();
            map_pts_.clear();
            for (const auto &kv : map.map_) {
                coords_.push_back(kv.first.x());
                coords_.push_back(kv.first.y());
                coords_.push_back(kv.first.z());
                start_.push_back(static_cast<int>(map_pts_.size() / 3));
                count_.push_back(static_cast<int>(kv.second.points.size()));
                for (const auto &p : kv.second.points) {
                    map_pts_.push_back(p.x());
                    map_pts_.push_back(p.y());
                    map_pts_.push_back(p.z());
                }
            }
        }
        const int n_vox = static_cast<int>(start_.size());
        const int n_pts = static_cast<int>(map_pts_.size() / 3);
        int rc = 0;
        {
            KISS_TIMER("gpu_set_map");
            rc = gpu_set_map(map.voxel_size_, n_vox, coords_.data(), start_.data(), count_.data(),
                             n_pts, map_pts_.data());
        }
        if (rc != 0) {
            Fail("gpu_set_map", rc);
            return false;
        }
        map_loaded_ = true;
        return true;
    }

    // Nearest-neighbour search for every source point. Returns false on failure.
    bool Associate(const std::vector<Eigen::Vector3d> &source,
                   const double max_distance,
                   Correspondences &out) {
        if (!map_loaded_) return false;  // SetMap already failed and was counted
        const std::size_t n = source.size();
        out.clear();
        if (n == 0) return true;
        {
            KISS_TIMER("gpu_flatten_source");
            src_.resize(3 * n);
            for (std::size_t i = 0; i < n; ++i) {
                src_[3 * i] = source[i].x();
                src_[3 * i + 1] = source[i].y();
                src_[3 * i + 2] = source[i].z();
            }
        }
        out_.resize(n);
        int rc = 0;
        {
            KISS_TIMER("gpu_run");
            rc = gpu_run(static_cast<int>(n), src_.data(), max_distance, out_.data());
        }
        if (rc != 0) {
            Fail("gpu_run", rc);
            return false;
        }
        {
            KISS_TIMER("gpu_build_pairs");
            out.reserve(n);
            for (std::size_t i = 0; i < n; ++i) {
                const int idx = out_[i];
                if (idx < 0) continue;
                const std::size_t k = 3 * static_cast<std::size_t>(idx);
                out.emplace_back(source[i],
                                 Eigen::Vector3d(map_pts_[k], map_pts_[k + 1], map_pts_[k + 2]));
            }
        }
        ::kiss_icp::prof::Count("gpu_iterations", 1.0);
        return true;
    }

    // =========================================================================
    // [NEW] Full ICP loop on the GPU
    // =========================================================================
    //
    // Intended use inside AlignPointsToMap, once SetMap succeeded:
    //
    //     SetSource(frame);                          // untransformed points
    //     Transform(n, initial_guess);               // same as TransformPoints(initial_guess)
    //     for (each iteration) {
    //         LoopIteration(n, max_distance, kernel_scale, JTJ, JTr, n_corr);
    //         CheckIteration(JTJ, JTr, n_corr, cpu_reference);   // only in check mode
    //         dx = JTJ.ldlt().solve(-JTr);  estimation = SE3d::exp(dx);
    //         Transform(n, estimation);               // the increment, not T_icp
    //     }
    //
    // If any call returns false, the points in device memory may be in an intermediate
    // state, so the caller rebuilds the source on the CPU before continuing there.

    // Uploads the source points once per scan.
    //
    // Receives: source, the scan points as Eigen vectors.
    // Returns : true on success; false if the loop is disabled, no map is loaded, the
    //           source is empty, or the upload failed (the last case is reported and
    //           counted in gpu_fallbacks).
    //
    // Flattens the points into the host buffer src_ and copies them to the device buffer
    // owned by gpu_associate (gpu_set_source). In verification mode it also keeps a host
    // copy that Transform will move in step with the GPU.
    bool SetSource(const std::vector<Eigen::Vector3d> &source) {
        source_loaded_ = false;
        n_src_ = 0;
        if (!loop_ready_ || !map_loaded_) return false;  // a failed SetMap was already counted
        if (source.empty()) return false;
        const std::size_t n = source.size();
        {
            KISS_TIMER("gpu_flatten_source");
            src_.resize(3 * n);
            for (std::size_t i = 0; i < n; ++i) {
                src_[3 * i] = source[i].x();
                src_[3 * i + 1] = source[i].y();
                src_[3 * i + 2] = source[i].z();
            }
        }
        int rc = 0;
        {
            KISS_TIMER("gpu_set_source");
            rc = gpu_set_source(static_cast<int>(n), src_.data());
        }
        if (rc != 0) {
            Fail("gpu_set_source", rc);
            return false;
        }
        if (check_) {
            KISS_TIMER("gpu_check_copy_source");
            check_source_ = source;
        }
        n_src_ = static_cast<int>(n);
        source_loaded_ = true;
        return true;
    }

    // Runs association and the normal equations for one ICP iteration on the GPU.
    //
    // Receives: n, number of source points (must equal the n of SetSource);
    //           max_distance, largest accepted correspondence distance; kernel_scale, the
    //           Geman-McClure scale; JTJ, JTr and n_corr, outputs.
    // Returns : true on success, false if the loop is not ready or a GPU call failed
    //           (reported and counted in gpu_fallbacks).
    //
    // gpu_run_device is asynchronous, so its timer measures only the launch. The
    // association kernel actually finishes inside gpu_normal_eq, whose blocking copy
    // waits for it: the "gpu_normal_eq" timer therefore includes both kernels plus the
    // download of the partial sums.
    //
    // The GPU returns the upper triangle of J^T w J. It is written to both triangles
    // because Eigen's ldlt() reads the lower one. n_corr comes from term 27, the number
    // of points with idx >= 0, the same value as correspondences.size() on the CPU.
    bool LoopIteration(int n,
                       double max_distance,
                       double kernel_scale,
                       Matrix6d &JTJ,
                       Vector6d &JTr,
                       int &n_corr) {
        if (!loop_ready_ || !source_loaded_) return false;
        if (n != n_src_) {
            Fail("LoopIteration (n differs from SetSource)", -2);
            return false;
        }
        int rc = 0;
        {
            KISS_TIMER("gpu_run_device");  // launch only: the call is asynchronous
            rc = gpu_run_device(n, max_distance);
        }
        if (rc != 0) {
            Fail("gpu_run_device", rc);
            return false;
        }
        double terms[GPU_ICP_NUM_TERMS];
        {
            KISS_TIMER("gpu_normal_eq");  // includes the association kernel (see above)
            rc = gpu_normal_eq(n, gpu_dev_src(), gpu_dev_pts(), gpu_dev_idx(), kernel_scale,
                               terms);
        }
        if (rc != 0) {
            Fail("gpu_normal_eq", rc);
            return false;
        }
        {
            KISS_TIMER("gpu_unpack_system");
            int o = 0;
            for (int i = 0; i < 6; ++i) {
                for (int j = i; j < 6; ++j) {
                    JTJ(i, j) = terms[o];
                    JTJ(j, i) = terms[o];  // symmetrize: ldlt() reads the lower triangle
                    ++o;
                }
            }
            for (int i = 0; i < 6; ++i) JTr(i) = terms[GPU_ICP_NUM_JTJ + i];
            // Term 27 is a sum of 1.0 values, so it holds an exact integer.
            n_corr = static_cast<int>(std::llround(terms[GPU_ICP_NUM_JTJ + GPU_ICP_NUM_JTR]));
        }
        ::kiss_icp::prof::Count("gpu_loop_iterations", 1.0);
        return true;
    }

    // Applies a pose to the source points in device memory, in place.
    //
    // Receives: n, number of source points (must equal the n of SetSource); T, the pose:
    //           initial_guess before the loop, and the increment SE3d::exp(dx) after each
    //           iteration (not the accumulated T_icp).
    // Returns : true on success, false if the loop is not ready or gpu_transform failed
    //           (reported and counted in gpu_fallbacks).
    //
    // Builds {qx, qy, qz, qw, tx, ty, tz} from T.unit_quaternion() and T.translation();
    // coeffs() of an Eigen quaternion is already in (x, y, z, w) order. In verification
    // mode it applies the same pose to the host copy with Sophus, exactly like
    // TransformPoints, so the CPU reference sees the same points as the GPU.
    bool Transform(int n, const Sophus::SE3d &T) {
        if (!loop_ready_ || !source_loaded_) return false;
        if (n != n_src_) {
            Fail("Transform (n differs from SetSource)", -2);
            return false;
        }
        double pose[GPU_POSE_SIZE];
        const auto &q = T.unit_quaternion().coeffs();  // (x, y, z, w)
        const Eigen::Vector3d &t = T.translation();
        pose[0] = q(0);
        pose[1] = q(1);
        pose[2] = q(2);
        pose[3] = q(3);
        pose[4] = t.x();
        pose[5] = t.y();
        pose[6] = t.z();
        int rc = 0;
        {
            KISS_TIMER("gpu_transform");
            rc = gpu_transform(n, gpu_dev_src(), pose);
        }
        if (rc != 0) {
            Fail("gpu_transform", rc);
            return false;
        }
        if (check_) {
            KISS_TIMER("gpu_check_transform");
            for (auto &p : check_source_) p = T * p;  // as TransformPoints in Registration.cpp
        }
        return true;
    }

    // Verification mode: compares one GPU linear system against the CPU one.
    //
    // Receives: JTJ, JTr and n_corr, the result of LoopIteration; cpu_reference, a
    //           callable that takes const std::vector<Eigen::Vector3d>& (the source points)
    //           and returns std::tuple<Matrix6d, Vector6d, int> with the CPU J^T w J,
    //           J^T w r and number of correspondences.
    // Returns : true if verification is off (nothing to do) or the comparison ran; false
    //           if it is on but there is no host copy of the source to compare with.
    //
    // The CPU reference is passed in because DataAssociation and BuildLinearSystem live in
    // an anonymous namespace of Registration.cpp, defined after this header is included.
    // From Registration.cpp it is a one-line lambda:
    //
    //     [&](const auto &src) {
    //         const auto c = DataAssociation(src, voxel_map, max_distance);
    //         const auto [A, b] = BuildLinearSystem(c, kernel_scale);
    //         return std::make_tuple(A, b, static_cast<int>(c.size()));
    //     }
    //
    // Recorded counters (one event per checked iteration):
    //   gpu_check_relerr_e15     max(|dJTJ|max / |JTJ_cpu|max, |dJTr|max / |JTr_cpu|max),
    //                            in units of 1e-15, because ProfileTimers prints counters
    //                            with %.0f; its "max" column is the worst iteration.
    //   gpu_check_count_mismatch 1 if n_corr differs from the CPU count, 0 otherwise; its
    //                            sum is the number of iterations that differ.
    //   gpu_check_count_absdiff  |n_corr - n_cpu|; its max is the largest difference.
    //
    // The error is not expected to be zero: the CPU sums with tbb::parallel_reduce, whose
    // order changes from run to run, and the GPU uses a fixed tree. Values around 1e-14
    // (10 in the counter) are rounding; a large error or a count mismatch is a real
    // difference.
    template <typename CpuReference>
    bool CheckIteration(const Matrix6d &JTJ,
                        const Vector6d &JTr,
                        int n_corr,
                        CpuReference &&cpu_reference) {
        if (!check_) return true;
        if (!source_loaded_ || check_source_.empty()) return false;
        KISS_TIMER("gpu_check");
        const auto ref = cpu_reference(static_cast<const std::vector<Eigen::Vector3d> &>(
            check_source_));
        const Matrix6d &A = std::get<0>(ref);
        const Vector6d &b = std::get<1>(ref);
        const int n_cpu = std::get<2>(ref);

        const double err_A =
            (JTJ - A).cwiseAbs().maxCoeff() / std::max(A.cwiseAbs().maxCoeff(), DBL_MIN);
        const double err_b =
            (JTr - b).cwiseAbs().maxCoeff() / std::max(b.cwiseAbs().maxCoeff(), DBL_MIN);
        const double rel = std::max(err_A, err_b);

        ::kiss_icp::prof::Count("gpu_check_relerr_e15", rel * 1e15);
        ::kiss_icp::prof::Count("gpu_check_count_mismatch", n_corr != n_cpu ? 1.0 : 0.0);
        ::kiss_icp::prof::Count("gpu_check_count_absdiff",
                                std::fabs(static_cast<double>(n_corr - n_cpu)));
        return true;
    }

private:
    Session() {
        const char *flag = std::getenv("KISS_USE_GPU");
        if (flag == nullptr || std::strcmp(flag, "1") != 0) return;
        const int rc = gpu_init(kMaxVoxels, kMaxMapPoints, kMaxSourcePoints);
        if (rc != 0) {
            std::fprintf(stderr, "gpu: gpu_init failed (code %d); running on the CPU\n", rc);
            return;
        }
        ready_ = true;

        // [NEW] Full GPU loop: needs KISS_USE_GPU=1 (above) and KISS_GPU_LOOP=1. The
        // normal-equation buffers are sized for the same maximum number of source points.
        if (EnvIsOne("KISS_GPU_LOOP")) {
            const int rc_icp = gpu_icp_init(kMaxSourcePoints);
            if (rc_icp != 0) {
                std::fprintf(stderr,
                             "gpu: gpu_icp_init failed (code %d); the ICP loop runs on the "
                             "CPU\n",
                             rc_icp);
            } else {
                loop_ready_ = true;
                // [NEW] Verification mode: only meaningful with the GPU loop.
                check_ = EnvIsOne("KISS_GPU_CHECK");
            }
        }
    }
    ~Session() {
        if (loop_ready_) gpu_icp_free();  // [NEW] reverse order of initialisation: gpu_icp first
        if (ready_) gpu_free();
    }

    static void Fail(const char *what, const int rc) {
        std::fprintf(stderr, "gpu: %s failed (code %d); this scan runs on the CPU\n", what, rc);
        ::kiss_icp::prof::Count("gpu_fallbacks", 1.0);
    }

    // [NEW] True if the environment variable exists and is exactly "1", the same rule as
    // KISS_USE_GPU.
    static bool EnvIsOne(const char *name) {
        const char *v = std::getenv(name);
        return v != nullptr && std::strcmp(v, "1") == 0;
    }

    bool ready_ = false;
    bool map_loaded_ = false;
    std::vector<int> coords_, start_, count_, out_;
    std::vector<double> map_pts_, src_;

    // [NEW] State of the full GPU loop.
    bool loop_ready_ = false;     // gpu_icp_init succeeded
    bool check_ = false;          // verification mode
    bool source_loaded_ = false;  // SetSource succeeded for the current map
    int n_src_ = 0;               // number of source points in device memory
    std::vector<Eigen::Vector3d> check_source_;  // host copy, only in verification mode
};

inline bool Enabled() { return Session::Get().enabled(); }
inline bool SetMap(const VoxelHashMap &map) { return Session::Get().SetMap(map); }
inline bool Associate(const std::vector<Eigen::Vector3d> &source,
                      const double max_distance,
                      Correspondences &out) {
    return Session::Get().Associate(source, max_distance, out);
}

// [NEW] Free-function wrappers for the full GPU loop, in the same style as above.
inline bool LoopEnabled() { return Session::Get().loop_enabled(); }
inline bool CheckEnabled() { return Session::Get().check_enabled(); }
inline bool SetSource(const std::vector<Eigen::Vector3d> &source) {
    return Session::Get().SetSource(source);
}
inline bool LoopIteration(int n,
                          double max_distance,
                          double kernel_scale,
                          Matrix6d &JTJ,
                          Vector6d &JTr,
                          int &n_corr) {
    return Session::Get().LoopIteration(n, max_distance, kernel_scale, JTJ, JTr, n_corr);
}
inline bool Transform(int n, const Sophus::SE3d &T) { return Session::Get().Transform(n, T); }
template <typename CpuReference>
inline bool CheckIteration(const Matrix6d &JTJ,
                           const Vector6d &JTr,
                           int n_corr,
                           CpuReference &&cpu_reference) {
    return Session::Get().CheckIteration(JTJ, JTr, n_corr,
                                         std::forward<CpuReference>(cpu_reference));
}

}  // namespace kiss_icp::gpu

#endif  // KISS_ICP_USE_GPU
