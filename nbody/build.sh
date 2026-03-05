#!/usr/bin/env bash
# ─── Quick single-command build (no CMake required) ──────────────────────────
# By default this script auto-detects your GPU's SM version via nvidia-smi.
# You can also pass the arch explicitly:  ./build.sh sm_75
# ─────────────────────────────────────────────────────────────────────────────
set -e

# ── Auto-detect GPU architecture ─────────────────────────────────────────────
if [ -n "$1" ]; then
    ARCH="$1"
else
    # nvidia-smi --query-gpu=compute_cap returns e.g. "7.5" for a T4
    RAW=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
          | head -1 | tr -d ' \r')
    if [ -n "$RAW" ]; then
        # "7.5" → "sm_75"
        ARCH="sm_$(echo "$RAW" | tr -d '.')"
        echo "Detected GPU compute capability: $RAW  →  $ARCH"
    else
        ARCH="sm_75"   # safe fallback (Turing / T4)
        echo "Could not detect GPU — defaulting to $ARCH"
    fi
fi

# ── Compile ───────────────────────────────────────────────────────────────────
echo "Compiling for architecture: $ARCH"
mkdir -p build
NVFLAGS="-O3 --use_fast_math -lineinfo -std=c++17 -arch=$ARCH -I src"
HOSTFLAGS="-O3 -ffast-math -march=native -fopenmp"

# CPU sources (shared by every binary) — compiled by the host compiler.
g++ -std=c++17 $HOSTFLAGS -c src/galaxy_ics.cpp -o build/galaxy_ics.o
g++ -std=c++17 $HOSTFLAGS -c src/cpu_nbody.cpp  -o build/cpu_nbody.o
CPU_OBJS="build/galaxy_ics.o build/cpu_nbody.o"

g++ -std=c++17 -O3 -fopenmp -I src src/bench_cpu.cpp $CPU_OBJS -o nbody_bench_cpu

nvcc $NVFLAGS -Xcompiler -fopenmp src/main.cu src/galaxy_simulation.cu \
     src/nbody_kernel.cu $CPU_OBJS -o galaxy_nbody
nvcc $NVFLAGS -Xcompiler -fopenmp src/bench_gpu.cu src/nbody_kernel.cu \
     $CPU_OBJS -o nbody_bench_gpu

echo ""
echo "Build successful → ./galaxy_nbody  ./nbody_bench_gpu  ./nbody_bench_cpu"
echo "Run   → ./galaxy_nbody [n_bodies] [n_steps] [--integrator kdk|euler]"
echo "Bench → ../scripts/run_benchmarks.sh"
echo "Plot  → python3 visualize.py simulation.bin [out.gif]"
