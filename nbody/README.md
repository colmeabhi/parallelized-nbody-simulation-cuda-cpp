# Galaxy N-Body Simulation — CUDA

A simplified Milky Way-like N-body gravitational simulation running on the GPU.

## What it simulates

| Component | Count (default) | Description |
|---|---|---|
| Central black hole | 1 | Sgr A* analogue, fixed mass |
| Disk stars | ~3 686 | Thin rotating disk, near-circular orbits |
| Bulge stars | ~409 | Dispersion-supported central bulge |
| **Total** | **4 096** | |

Physics: direct O(N²) gravity, leapfrog integration, G = 1 (normalised units).

---

## File structure

```
nbody/
├── src/
│   ├── nbody_kernel.cuh       ← GPU solver interface
│   ├── nbody_kernel.cu        ← CUDA kernels: tiled force calc + integration
│   ├── galaxy_simulation.cuh  ← Galaxy simulation interface
│   ├── galaxy_simulation.cu   ← Initial conditions + time-loop (no CUDA API)
│   └── main.cu                ← Entry point
├── CMakeLists.txt
├── build.sh                   ← One-liner build without CMake
├── visualize.py               ← Python animation of output
└── README.md
```

**Separation of concerns:**
- `nbody_kernel.*` — knows about CUDA (tiles, shared memory, kernels).
- `galaxy_simulation.*` — knows about galaxies (disk setup, circular orbits).
  It calls only the three `NBodyGPU` methods: `allocate`, `step`, `download`.

---

## Build

### Option A — CMake (recommended)

```bash
cd nbody
mkdir build && cd build
cmake .. && make -j$(nproc)
./galaxy_nbody
```

### Option B — direct nvcc

```bash
cd nbody
chmod +x build.sh
./build.sh sm_86     # replace sm_86 with your GPU arch (see below)
```

**Common GPU architectures:**

| GPU | `-arch` |
|---|---|
| GTX 10xx (Pascal) | `sm_60` |
| V100 (Volta) | `sm_70` |
| RTX 20xx (Turing) | `sm_75` |
| RTX 30xx / A100 (Ampere) | `sm_80` or `sm_86` |
| RTX 40xx (Ada) | `sm_89` |

---

## Run

```bash
./galaxy_nbody                   # defaults: 4096 bodies, 5000 steps
./galaxy_nbody 8192 10000        # custom: 8192 bodies, 10000 steps
```

This writes **`simulation.bin`** (≈ 5 MB for defaults).

---

## Visualize

```bash
pip install numpy matplotlib
python3 visualize.py simulation.bin          # live animation window
python3 visualize.py simulation.bin out.gif  # save to GIF
```

The animation shows a top-down (X-Y) view of the galaxy.
The central black hole is the gold ★ marker.

---

## How the CUDA kernel works (brief)

```
for each tile of BLOCK_SIZE bodies:
    load tile into shared memory          (1 global read per thread)
    for each body k in tile:
        acc += gravity(my_body, tile[k]) (reads from L1/shared memory)
    __syncthreads()

velocity += acc * dt   // Kick
position += vel * dt   // Drift
```

Shared-memory tiling gives a **BLOCK_SIZE× reduction** in global-memory reads
compared to a naïve kernel, which is the main performance optimisation.

---

## Tuning parameters (`GalaxyConfig` in `galaxy_simulation.cuh`)

| Parameter | Default | Meaning |
|---|---|---|
| `n_bodies` | 4096 | total body count |
| `bh_mass_frac` | 0.05 | BH mass / total mass |
| `bulge_frac` | 0.10 | fraction of stars in bulge |
| `disk_radius` | 1.0 | outer disk edge (sim units) |
| `disk_thickness` | 0.02 | vertical scale height |
| `dt` | 0.01 | timestep (v_c ≈ 1, outer orbit ≈ 6.3 time units) |
| `n_steps` | 5000 | integration steps |
| `out_interval` | 50 | steps between saved snapshots |

> **Stability tip:** if stars escape early, reduce `dt` or increase `SOFTENING`
> in `nbody_kernel.cuh`.
