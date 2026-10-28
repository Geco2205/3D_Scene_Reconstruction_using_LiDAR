# Vendored KISS-ICP

Modified copy of the C++ core of [PRBonn/kiss-icp](https://github.com/PRBonn/kiss-icp)
(`cpp/kiss_icp`), MIT license (see `LICENSE` in this folder).

* Base: tag `v1.0.0`, commit `345d0b42a7d2f815c80bd882d2324d7768ad5273`.
* Changes made for this project (4 files; full diff in `docs/kiss-icp-changes.diff`):

| File | Change |
|---|---|
| `core/ProfileTimers.hpp` (new) | per-stage timers and counters (`KISS_TIMER`, `prof::Count`) |
| `core/Registration.cpp` | timers around ICP stages; optional GPU hook (`#ifdef KISS_ICP_USE_GPU`) |
| `pipeline/KissICP.cpp` | timers around the pipeline stages |
| `core/GpuAssociation.hpp` (new) | GPU offload of `DataAssociation` (only with `RECON_GPU=ON`) |

* The original `DataAssociation` is untouched and is the CPU path.
* Env vars: `KISS_PROFILE_OUT=<file>` writes the timers/counters CSV; `KISS_USE_GPU=1` enables the GPU.
* Do not apply upstream patches blindly on top of this tree; compare with the diff above.
