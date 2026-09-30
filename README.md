# 3D_Scene_Reconstruction_using_LiDAR# Scene Reconstruction using LiDAR

CPU prototype of a LiDAR scene reconstruction pipeline, developed for the course
EL5859 Heterogeneous Computing (Instituto Tecnológico de Costa Rica, 2026).
It turns a sequence of LiDAR scans into a 3D triangle mesh of the environment.

```
.pcd scans → range filter (NEON/scalar) → KISS-ICP (pose) → VDBFusion (TSDF) → .ply mesh
```


This is the CPU baseline of the project. Later stages will offload the bottlenecks
found here to a GPU (NVIDIA Jetson Nano, CUDA) and an FPGA (AMD Kria KV260, HLS).

## Youtube Video

This youtube video showcase a little demostration of the pipeline working a seeing the mesh at meshlab.

https://youtu.be/nKCmF2cM_jg

## What is reused and what we wrote

| Component | Origin | License |
|---|---|---|
| 3D registration (LiDAR odometry) | [KISS-ICP](https://github.com/PRBonn/kiss-icp) `v1.0.0`, used as a library | MIT |
| TSDF integration and mesh extraction | [VDBFusion](https://github.com/PRBonn/vdbfusion) `v0.1.6`, used as a library | MIT |
| Pipeline (`src/main.cpp`) | Written for this project, based on VDBFusion's `examples/cpp/kitti_pipeline.cpp` | — |
| PCD reader (`src/PcdReader.cpp`) | Written for this project, following the [PCD file format specification](https://pointclouds.org/documentation/tutorials/pcd_file_format.html) | — |
| Range filter with ARM NEON (`src/RangeFilter.cpp`) | Written for this project | — |
| Scan ordering, PLY writer, timer (`src/Io.cpp`) | Written for this project | — |
| Synthetic scan generator (`tools/make_synthetic_scans.py`) | Written for this project | — |




## Dependencies

Fedora:

```bash
sudo dnf install gcc-c++ cmake git eigen3-devel tbb-devel openvdb-devel \
                 imath-devel boost-devel blosc-devel python3-numpy
```

Ubuntu (including the Kria):

```bash
sudo apt install build-essential cmake git libeigen3-dev libtbb-dev \
                 libopenvdb-dev libboost-iostreams-dev libblosc-dev python3-numpy
```

OpenVDB must come from the system package manager. If CMake cannot find it, the
configuration stops with an error instead of trying to build it from source.

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

The first configuration needs internet access to download KISS-ICP and VDBFusion.

## Quick test (no dataset needed)

```bash
python3 tools/make_synthetic_scans.py --out data/synthetic --scans 20
./build/recon --scans data/synthetic --out synthetic.ply
```

Expected: the program prints the mesh size and the average time per stage, and
writes `synthetic.ply`. This only checks that everything builds and runs; the
synthetic scene is too simple to evaluate registration accuracy.

## Data

We use the [Newer College Dataset](https://ori-drs.github.io/newer-college-dataset/)
(2020, Ouster OS1-64 at 10 Hz), sequence 01 "short experiment".

### Option A: sample (300 scans, ~30 s of walking)

A ready-to-use sample with the matching ground-truth poses is attached to the
[v1.0.0 release](https://github.com/Geco2205/3D_Scene_Reconstruction_using_LiDAR/releases/tag/v1.0.0):

```bash
wget https://github.com/Geco2205/3D_Scene_Reconstruction_using_LiDAR/releases/download/v0.1.0/ncd_sample.zip
unzip ncd_sample.zip -d data
./build/recon --scans data/ncd_sample/scans --out sample.ply --icp-voxel 1.0
```

### Option B: full dataset

1. Request access through the form on the dataset website.
2. In `2020-ouster-os1-64-realsense`, open the folder of sequence 01 (short
   experiment) and download `raw_format/ouster_zip_files/` (10 zips)
   and `ground_truth/`.
3. Unzip all 10 files into the same folder. The scans are spread across the zips
   without temporal order, so all of them are needed to get a continuous sequence:

```bash
mkdir -p data/ncd
for z in ouster_scan-*.zip; do unzip -jq "$z" -d data/ncd; done
rm -f data/ncd/*"(1)".pcd    # one file is duplicated in the zips
ls data/ncd | wc -l           # expected: 15301
```

## Usage

```bash
./build/recon --scans <dir> [options]
```

| Option | Default | Description |
|---|---|---|
| `--scans <dir>` | required | Directory with the `.pcd` scans |
| `--out <file>` | `mesh.ply` | Output mesh |
| `--max-scans <n>` | `0` (all) | Number of scans to process |
| `--tsdf-voxel <m>` | `0.10` | TSDF voxel size |
| `--icp-voxel <m>` | `0.50` | KISS-ICP voxel size |
| `--min-range <m>` | `1.0` | Minimum point range |
| `--max-range <m>` | `60.0` | Maximum point range |
| `--timing-csv <file>` | — | Write per-stage timings to a CSV file |
| `--pipeline <mode>` | `off` | Task parallelism: `off`, `prefetch` or `full` (see [Optimizations](#optimizations)) |

Scans are processed in timestamp order. File names follow
`cloud_<sec>_<nsec>.pcd` and are sorted numerically, because the nanosecond
field does not always have 9 digits.

## Outputs

- **`<out>.ply`**: triangle mesh (ASCII PLY). Open it with MeshLab or CloudCompare.
- **`<timing-csv>`**: one row per scan. For each stage (`read`, `filter`,
  `convert`, `register`, `transform`, `integrate`) there is a wall-clock column
  `<stage>_ms` and a process CPU-time column `<stage>_cpu_ms`, plus
  `scan, points_in, points_kept` and `rss_kb` (resident memory after the scan).
- **`<mesh-csv>`**: one row per mesh extraction with
  `repeat, mesh_ms, mesh_cpu_ms, vertices, triangles`.

## Viewing the mesh

We use [MeshLab](https://www.meshlab.net/) to inspect the reconstruction.

Fedora:

```bash
sudo dnf install meshlab
```

Ubuntu:

```bash
sudo apt install meshlab
```

Open the mesh:

```bash
meshlab sample.ply
```

If the 3D view is blank (common on Wayland, the default on Fedora and recent
Ubuntu), launch it with:

```bash
QT_QPA_PLATFORM=xcb meshlab sample.ply
```
  

### Per-stage instrumentation

The pipeline is instrumented so that each stage is measured with two clocks:
wall time (`steady_clock`) and process CPU time (`CLOCK_PROCESS_CPUTIME_ID`,
summed over all threads). Their ratio is the average number of cores the stage
used, which is what tells a stage that is slow apart from one that is already
parallel. Memory is read from `/proc/self/statm` at the end of every scan, and
the process peak from `getrusage`.

The stages are `read`, `filter`, `convert`, `register`, `transform` and
`integrate`. `transform` (mapping the points into the global frame) is measured
separately from the TSDF integration: in the original code that loop sat inside
the integration timer and inflated it.

#### Running it

```bash
tools/run_profile.sh --label pc1-nicole --scans data/ncd_sample/scans --icp-voxel 1.0
```

Plug in laptops and close other programs first, because on battery the
processor lowers its frequency and the times stop being comparable. On a
desktop CPU it takes about 3 minutes; on the Kria or the Jetson it takes longer,
mostly because of the 101 mesh repetitions.

It writes `results/<label>/` with `timings.csv`, `mesh.csv`, `system.txt` and
`run.log`. Mesh extraction happens once per sequence, so it is repeated on the
final volume (`--mesh-repeats`, 101 by default) to reach more than 100 samples
per stage, as the project requires. The 300-scan sample gives 300 samples for
every per-scan stage.

`system.txt` records the arguments, compiler, build flags, commit, CPU model
and thread count, so that a difference between machines can be explained rather
than guessed.

#### Analyzing it

```bash
python3 tools/analyze_profile.py results
```

It reads every `results/*/timings.csv` and writes to `results/analysis/`:

| File | Content |
|---|---|
| `summary.md` | Per machine: n, mean, std, 95 % CI, min, p50, p95, max, share of the scan time and CPU/wall ratio for each stage, plus memory and throughput. A cross-machine comparison table when there is more than one machine. |
| `summary.csv` | The same statistics in tabular form |
| `fig_breakdown.png` | Mean time per scan split by stage, one bar per machine |
| `fig_stages_box.png` | Distribution of every stage per machine (log scale) |
| `fig_timeline_<label>.png` | Time per scan and resident memory along the sequence |

`--skip-first N` drops the first `N` scans of every machine as warm-up (the
first registrations are cheaper because the local map is still empty).

The script checks that every stage has more than 100 samples and reports which
ones fall short, since with fewer samples the confidence interval of the mean
is not useful.


## Optimizations

The optimizations were chosen and applied following the course material,
mainly Chapter 2 (*Sistemas multiprocesador y su programación*). The chapter
gave both the method (measure the baseline, find where the time goes, optimize,
measure again) and the techniques: task parallelism with producer-consumer
queues and ping-pong buffers, compiler options (`-march=native`, LTO), and data
layout as a structure of arrays aligned to the cache line.

Each optimization can be switched on or off independently, so its effect is
always measured against the same baseline. With the default build and
`--pipeline off`, the program is the baseline.

| Switch | Where | Default | Effect |
|---|---|---|---|
| `RECON_OPT_NATIVE` | CMake | `OFF` | `-march=native` (x86) / `-mcpu=native` (ARM), also applied to KISS-ICP and VDBFusion |
| `RECON_OPT_LTO` | CMake | `OFF` | Link-Time Optimization (`-flto`) |
| `RECON_OPT_SOA_ALIGNED` | CMake | `OFF` | Point arrays aligned to 64 bytes and branch-free compaction in the reader and the filter |
| `RECON_OPT_NEON` | CMake | `ON` | NEON intrinsics in the range filter (ARM only) |
| `--pipeline` | runtime | `off` | `prefetch` or `full` task parallelism, described below |

### How it is measured

`tools/run_optimizations.sh` builds every variant, runs each one on the 300-scan
sample through `tools/run_profile.sh` and writes the comparison table to
`results/opt-<label>/analysis/summary.md`:

```bash
tools/run_optimizations.sh --label pc-<name> --scans data/ncd_sample/scans --icp-voxel 1.0
```

Each variant changes one thing with respect to `base`, except `all`, which
combines everything. Laptops must stay plugged in, with no other heavy programs
open, because on battery the CPU lowers its frequency and the variants stop
being comparable. With `--pipeline` the stages overlap, so the sum of the stage
times is no longer the time per scan: throughput is computed from the measured
period between finished scans (`done_ms` column), and the speedup is taken
against `base` on the same machine.

### Where the time goes

In the baseline of the three machines measured so far, registration (KISS-ICP)
and TSDF integration (VDBFusion) take between 96 % and 98 % of the time per scan,
while reading, filtering and converting the points take less than 2 % (per-stage
times in [Expected results](#expected-results)):

| Machine | Register + integrate | Read + filter + convert |
|---|---:|---:|
| PC1 (i3-1115G4) | 97.1 % | 1.3 % |
| PC2 (i5-1135G7) | 96.3 % | 1.7 % |
| Kria KV260 (Cortex-A53) | 97.8 % | 1.4 % |

By Amdahl's law, speeding up only the reader and the filter can improve the
whole scan by less than 2 %. The optimizations that can matter are the ones
that reach the libraries or that run the two heavy stages at the same time.

### Applied

**Task parallelism (`--pipeline`).** Load, register and integrate are organized as a producer-consumer pipeline, each stage in its own thread, connected by bounded FIFO queues of capacity 2, following a scheme similar to the ping-pong buffers of Chapter 2. The scans are registered and integrated in the same order as in the sequential run; on PC2 the mesh was verified to have the same number of vertices and triangles as the baseline in both pipeline modes. `prefetch` only hides the load stage, which is less than 2 % of the time, and gives 1.00×. `full` also overlaps registration of scan *k+1* with integration of scan *k* and gives **1.48×** on PC2 (period 50.8 → 34.3 ms). Integration runs on a single core (CPU/wall of 1.00 on the three machines), and with `full` the period (34.27 ms) is almost equal to the integration time (34.18 ms), so integration is now the bottleneck of the pipeline.

**`-march=native` / `-mcpu=native`.** It lets the compiler use the CPU's extensions (AVX2/FMA, Cortex-A tuning), also in KISS-ICP and VDBFusion. It gives **1.03×** on PC2. The limited gain is consistent with OpenVDB and TBB being precompiled dependencies that these flags do not reach, although this test does not attribute the difference to that cause alone. The binary is not portable, and the mesh differs from the baseline by 3 triangles out of 1 476 686, attributable to changes in floating-point code generation; FMA was not isolated as the only cause.

**LTO.** It allows inlining across files and across the libraries built from source. No measurable improvement was observed (**1.00×**).

**Aligned SoA and branch-free compaction.** The point arrays are aligned to 64 bytes (one cache line) and written by index without branches. `read` goes from 0.54 to 0.46 ms and `filter` from 0.18 to 0.09 ms, but the overall result is **0.98×**, so no improvement of the total time was observed, as Amdahl's law predicts for stages that take less than 2 % of the time.

**NEON in the range filter.** It was already in the prototype; the new switch only turns it off (`noneon`) so its effect can be measured on the Kria or the Jetson.

### Not applied

Thread affinity was not applied because the profiling did not show evidence that thread migrations were a bottleneck, and part of the parallelism is managed internally by external libraries (TBB in KISS-ICP). NUMA-aware allocation does not apply because every target machine has a single socket.

`-ffast-math` and `-Ofast` were discarded because, as the course notes warn, they can change numerical results and the handling of special cases such as NaN and infinity, which the reader relies on to drop invalid points. PGO was not applied because of the extra cost of generating a representative profile and rebuilding, prioritizing more direct and reproducible optimizations for this prototype.

KISS-ICP and VDBFusion were not modified with manual intrinsics because the profiling identifies hotspots at the stage level but does not yet show which internal loops would justify manual SIMD, and the course material recommends trying autovectorization, checking the compiler reports and measuring before resorting to intrinsics; those stages are left for the GPU and FPGA prototypes. The SoA layout was not extended past the filter because both libraries take `Eigen::Vector3d` arrays in their API, and mesh extraction was not parallelized because it runs once per sequence and does not affect throughput per scan.

## Expected results

### Without optimizations (baseline)

Mean time per stage, in ms per scan, over the 300-scan sample with the default
build and `--pipeline off`:

| Stage | PC1 (i3-1115G4) | PC2 (i5-1135G7) | Kria KV260 (Cortex-A53) |
|---|---:|---:|---:|
| read | 0.72 | 0.54 | 4.99 |
| filter | 0.26 | 0.18 | 2.16 |
| convert | 0.16 | 0.14 | 0.84 |
| register | 45.67 | 21.32 | 305.09 |
| transform | 1.34 | 0.99 | 4.35 |
| integrate | 38.50 | 27.57 | 249.27 |
| **Total per scan** | **86.65** | **50.74** | **566.71** |

Sources: `results/pc1-nicole/`, `results/opt-pc2-keilin/base/` and
`results/kria-gonzalo/`.

### With optimizations

Throughput in scans/s over the 300-scan sample, with the speedup against `base`
on the same machine (from the `Ciclo de scans` line of
`results/opt-<label>/<variant>/run.log`):

| Variant | PC1 (i3-1115G4) | PC2 (i5-1135G7) | Kria KV260 |
|---|---:|---:|---:|
| base | 11.75 (1.00×) | 19.68 (1.00×) | 1.72 (1.00×) |
| prefetch | 11.81 (1.00×) | 19.70 (1.00×) | 1.75 (1.01×)|
| full | 14.07 (1.20×) | 29.17 (1.48×) | 2.51 (1.46×)|
| native | 12.09 (1.03×) | 20.33 (1.03×) | 1.73 (1.00×)|
| lto | 11.56 (0.98×) | 19.67 (1.00×) | 1.74 (1.01×)|
| soa | 12.12 (1.03×) | 19.37 (0.98×) | 1.75 (1.02×)|
| all | 17.93 (1.53×) | 29.06 (1.48×) | 2.16 (1.25×)|
| noneon (ARM only) | — | — | 1.73 (1.00×)|

> **Kria baseline note.** The first `base` run showed anomalous file-I/O
> behavior in the `read` stage. The variant was repeated under the same
> configuration, obtaining 1.72 scans/s. This rerun is used as the reference
> for the Kria optimization speedups reported above.

The earlier baseline profiling and the optimization matrix were collected in
different revisions of the prototype. Therefore, the optimization speedups
reported here are computed against the `base` variant from the same optimization
experiment.

On the Kria KV260, the `full` task-parallel pipeline increased throughput
from 1.72 to 2.51 scans/s (**1.46×**). The combined `all` variant reached
2.16 scans/s (**1.25×**), showing that enabling every optimization together
did not outperform the task-parallel pipeline by itself on this platform.






## Next steps

The next steps are to implement the TSDF integration in CUDA on the Jetson Nano,
since each point updates its voxels independently, and to accelerate the
nearest-neighbor search of the registration, either on the GPU or on an HLS
pipeline on the Kria KV260, given that it is the stage most affected by the
embedded platform's limited core count.



## Usage of AI

Note on the use of AI tools

AI tools are used as support to understand concepts, generate ideas, and improve the writing of the documentation. Implementation, validation, and results are the student's responsibility, who assumes responsibility for any use beyond, or not described in, the above.

The shared conversation links are attached as evidence.

Gerson: https://claude.ai/share/be045057-e15d-462e-9e98-40855a3e21fa

Nicole: https://chatgpt.com/share/6ab9b72b-0e5c-83e8-983b-1d71080f1d52

Keilin: https://chatgpt.com/share/6abbee3d-ba9c-83e8-bdcb-177f0710a5cb

Gonzalo: https://chatgpt.com/share/6abc539e-4118-83e8-bd86-a0ebd38bcdda


## References

- I. Vizzo, T. Guadagnino, B. Mersch, L. Wiesmann, J. Behley, and C. Stachniss,
  "KISS-ICP: In Defense of Point-to-Point ICP – Simple, Accurate, and Robust
  Registration If Done the Right Way," *IEEE Robotics and Automation Letters*,
  vol. 8, no. 2, pp. 1029–1036, 2023.
- I. Vizzo, T. Guadagnino, J. Behley, and C. Stachniss, "VDBFusion: Flexible and
  Efficient TSDF Integration of Range Sensor Data," *Sensors*, vol. 22, no. 3,
  p. 1296, 2022.
- M. Ramezani, Y. Wang, M. Camurri, D. Wisth, M. Mattamala, and M. Fallon,
  "The Newer College Dataset: Handheld LiDAR, Inertial and Vision with Ground
  Truth," in *Proc. IEEE/RSJ IROS*, 2020. Licensed under CC BY-NC-SA 4.0.

## Authors

Gonzalo Alpízar Salas, Gerson Adrián Cordero Zúñiga, Nicole Irina Corrales
Rodríguez, Keilin Loáisiga Téllez — Escuela de Ingeniería Electrónica, ITCR.
