#!/usr/bin/env bash
# ─── Quick single-command build (no CMake required) ──────────────────────────
# Adjust -arch to match your GPU.  Run `nvidia-smi` to find your architecture:
#   SM 6.x  →  -arch=sm_60  (Pascal,  GTX 10xx)
#   SM 7.0  →  -arch=sm_70  (Volta,   V100)
#   SM 7.5  →  -arch=sm_75  (Turing,  RTX 20xx)
#   SM 8.0  →  -arch=sm_80  (Ampere,  A100 / RTX 30xx)
#   SM 8.6  →  -arch=sm_86  (Ampere,  RTX 3070-3090)
#   SM 8.9  →  -arch=sm_89  (Ada,     RTX 40xx)
# ─────────────────────────────────────────────────────────────────────────────
set -e

ARCH=${1:-sm_86}    # pass your arch as first arg, e.g.: ./build.sh sm_75

echo "Building for architecture: $ARCH"

nvcc -O3 --use_fast_math -std=c++14 \
     -arch="$ARCH"                  \
     -I src                         \
     src/main.cu                    \
     src/galaxy_simulation.cu       \
     src/nbody_kernel.cu            \
     -o galaxy_nbody

echo "Done → ./galaxy_nbody"
echo "Run  → ./galaxy_nbody [n_bodies] [n_steps]"
echo "Plot → python3 visualize.py simulation.bin"
