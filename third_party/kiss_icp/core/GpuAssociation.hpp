// GpuAssociation.hpp - optional GPU offload of DataAssociation (library gpu_assoc).
//
// Compiled only when KISS_ICP_USE_GPU is defined. At runtime it is enabled with the
// environment variable KISS_USE_GPU=1; otherwise Enabled() is false and nothing changes.
//
// The local map does not change during the ICP loop, so it is flattened and uploaded once
// per scan (SetMap), and every iteration only sends the source points (Associate).
// If any GPU call fails, the error is reported on stderr, counted in "gpu_fallbacks", and
// that scan finishes on the CPU, so a GPU problem never silently changes the result.
#pragma once

#ifdef KISS_ICP_USE_GPU

#include <Eigen/Core>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "ProfileTimers.hpp"
#include "VoxelHashMap.hpp"
#include "gpu_associate.h"

namespace kiss_icp::gpu {

using Correspondence = std::pair<Eigen::Vector3d, Eigen::Vector3d>;
using Correspondences = std::vector<Correspondence>;

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

    // Flattens the voxel map and uploads it. Returns false if the scan must use the CPU.
    bool SetMap(const VoxelHashMap &map) {
        map_loaded_ = false;
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
    }
    ~Session() {
        if (ready_) gpu_free();
    }

    static void Fail(const char *what, const int rc) {
        std::fprintf(stderr, "gpu: %s failed (code %d); this scan runs on the CPU\n", what, rc);
        ::kiss_icp::prof::Count("gpu_fallbacks", 1.0);
    }

    bool ready_ = false;
    bool map_loaded_ = false;
    std::vector<int> coords_, start_, count_, out_;
    std::vector<double> map_pts_, src_;
};

inline bool Enabled() { return Session::Get().enabled(); }
inline bool SetMap(const VoxelHashMap &map) { return Session::Get().SetMap(map); }
inline bool Associate(const std::vector<Eigen::Vector3d> &source,
                      const double max_distance,
                      Correspondences &out) {
    return Session::Get().Associate(source, max_distance, out);
}

}  // namespace kiss_icp::gpu

#endif  // KISS_ICP_USE_GPU
