# Galaxy N-Body Simulation — CUDA + CPU baselines

A simplified Milky Way-like N-body gravitational simulation, with a CUDA
solver (naive and shared-memory tiled kernels) and CPU reference solvers
(single-thread, OpenMP, Barnes–Hut).

## What it simulates

| Component | Count (default) | Description |
|---|---|---|
| Central black hole | 1 | Sgr A* analogue, 5 % of total mass |
| Disk stars | 3 686 | Thin rotating disk, near-circular orbits |
| Bulge stars | 409 | Dispersion-supported central bulge |
| **Total** | **4 096** | |

Physics: direct O(N²) gravity with Plummer softening ε = 0.02,
**kick-drift-kick leapfrog** integration, G = 1, total mass 1, disk radius 1.

---

## Build

### CMake (recommended)

```bash
cd nbody
cmake -S . -B build
cmake --build build -j
```

This produces:

| Binary | Needs CUDA | Purpose |
|---|---|---|
| `build/galaxy_nbody` | yes | the galaxy simulation → `simulation.bin`, `energy.csv` |
| `build/nbody_bench_gpu` | yes | one benchmark run of a GPU kernel → CSV row |
| `build/nbody_bench_cpu` | no | one benchmark run of a CPU solver → CSV row |

Without `nvcc` only `nbody_bench_cpu` is built. The CUDA architecture defaults
to the GPU in the machine (`native`, CMake ≥ 3.24). To override it:
`-DCMAKE_CUDA_ARCHITECTURES="75;86"`.

### Without CMake

```bash
cd nbody
./build.sh            # auto-detects the GPU's sm_XX, or: ./build.sh sm_86
```

| GPU | arch |
|---|---|
| RTX 20xx / T4 (Turing) | `sm_75` |
| A100 | `sm_80` |
| RTX 30xx (Ampere) | `sm_86` |
| RTX 40xx / L4 (Ada) | `sm_89` |
| H100 (Hopper) | `sm_90` |

CUDA 13 dropped Pascal (`sm_60`) and Volta (`sm_70`). Use CUDA 12 for those.

---

## Run

```bash
./build/galaxy_nbody                          # 4096 bodies, 5000 steps
./build/galaxy_nbody 8192 10000               # positional: n_bodies n_steps
./build/galaxy_nbody --dt 0.0025 --steps 20000
./build/galaxy_nbody --integrator euler       # old scheme, for comparison
./build/galaxy_nbody --kernel naive --block 128
```

At the end of the run, the program prints the energy check:

```
  Energy  : E0 = …   E_end = …
            final dE/E0 = …   max |dE/E0| = …
```

`energy.csv` holds the kinetic, potential and total energy at every snapshot.

### Benchmark binaries

```bash
./build/nbody_bench_gpu --header --kernel tiled --n 65536 --block 256 --steps 20 --energy
./build/nbody_bench_cpu --impl omp --n 16384 --steps 5
./build/nbody_bench_cpu --impl bh --n 65536 --theta 0.5 --force-error
./build/nbody_bench_cpu --impl omp --n 1024 --dt 0.005 --steps 4000 \
                        --integrator kdk --energy-log e.csv --energy-every 40
```

CSV columns: `impl, integrator, n, block, threads, steps, ms_per_step,
interactions_per_s, gflops, energy_drift, force_rel_err`.
`../scripts/run_benchmarks.sh` runs the full sweep.

---

## Visualize

```bash
pip install numpy matplotlib
python3 visualize.py simulation.bin          # live animation window
python3 visualize.py simulation.bin out.gif  # save to GIF
```

---

## How the GPU step works

```
kick_drift_kernel   (1 thread per body, touches only body i)
    v_i += a_i · dt/2
    x_i += v_i · dt
── kernel boundary: every block is done before any block reads x ──
accel_tiled_kernel  (1 thread per body)
    for each tile of blockDim.x bodies:
        tile[threadIdx.x] = x[tile_start + threadIdx.x]   // 1 global load/thread
        __syncthreads()
        for k in tile:  a_i += m_k (x_k − x_i) / (|x_k − x_i|² + ε²)^{3/2}
        __syncthreads()
    acc[i] = a_i
    v_i  += a_i · dt/2
```

`accel_naive_kernel` is the same loop reading `x[j]` straight from global
memory. Every thread in a warp reads the same `x[j]` at the same time, so that
load is a broadcast that L1 caches well. The tiled kernel's advantage comes
from fewer L1/L2 transactions and from not depending on cache hit rates.
Nsight Compute (`scripts/profile_ncu.sh`) measures how large that advantage
actually is on a given GPU.

The original version did the force and the position update in **one**
kernel. Other blocks could still be reading `pos[]` while it was being
overwritten, which is a data race. See the top-level README for details.

---

## Parameters

`SimConfig` / `GalaxyConfig` (`galaxy_simulation.cuh`, `galaxy_ics.h`):

| Parameter | Default | Meaning |
|---|---|---|
| `n_bodies` | 4096 | total body count |
| `bh_mass_frac` | 0.05 | BH mass / total mass |
| `bulge_frac` | 0.10 | fraction of bodies in the bulge |
| `disk_radius` | 1.0 | outer disk edge (sim units) |
| `disk_thickness` | 0.02 | vertical scale height |
| `dt` | 0.01 | timestep (outer orbit ≈ 6.3 time units) |
| `n_steps` | 5000 | integration steps |
| `out_interval` | 50 | steps between saved snapshots |

`kSoftening` (ε) is in `common.h`.

> **Accuracy tip:** at the default `dt = 0.01`, total energy drifts by a few
> percent over ~20 time units. Stars passing close to the softened black hole
> are under-resolved (orbital time ≈ √(ε³/M_bh) ≈ 0.013). Use `--dt 0.0025`
> for about 10⁻⁴ energy error, or increase ε.
