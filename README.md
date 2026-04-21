# Parallel Programming Project

GPU-accelerated N-body gravitational simulation of a Milky Way-like galaxy using CUDA.

## Project Structure

```
├── nbody/                  ← Galaxy N-body simulation (CUDA)
│   ├── src/
│   │   ├── main.cu                 Entry point
│   │   ├── nbody_kernel.cu/.cuh    CUDA kernels (tiled force calc + integration)
│   │   └── galaxy_simulation.cu/.cuh   Initial conditions + time loop
│   ├── CMakeLists.txt
│   ├── build.sh            Quick build without CMake
│   ├── visualize.py        Python animation of output
│   └── README.md           Detailed docs (build, run, tuning)
└── README.md               ← You are here
```

## Quick Start

```bash
cd nbody
mkdir build && cd build
cmake .. && make -j$(nproc)
./galaxy_nbody              # 4096 bodies, 5000 steps
```

See [`nbody/README.md`](nbody/README.md) for full build options, GPU architecture flags, and visualization instructions.
