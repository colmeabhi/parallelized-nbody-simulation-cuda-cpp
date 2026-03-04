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

nvcc -O3 --use_fast_math -std=c++14 \
     -arch="$ARCH"                  \
     -I src                         \
     src/main.cu                    \
     src/galaxy_simulation.cu       \
     src/nbody_kernel.cu            \
     -o galaxy_nbody

echo ""
echo "Build successful → ./galaxy_nbody"
echo "Run  → ./galaxy_nbody [n_bodies] [n_steps]"
echo "Plot → python3 visualize.py simulation.bin [out.gif]"
