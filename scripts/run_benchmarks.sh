#!/usr/bin/env bash
# run_benchmarks.sh  —  Build everything and collect all benchmark data into
#                       results/*.csv.  Then run scripts/make_report.py.
#
#   scripts/run_benchmarks.sh              # CPU + GPU (GPU skipped if absent)
#   SKIP_CPU=1 scripts/run_benchmarks.sh   # GPU only (e.g. on a cluster node)
#   SKIP_GPU=1 scripts/run_benchmarks.sh   # CPU only
#   QUICK=1    scripts/run_benchmarks.sh   # fewer steps, for a smoke test
#
# Environment overrides: BUILD_DIR, RESULTS_DIR, CUDA_ARCH (e.g. 86),
#                        NS (N list), BLOCKS (block-size list), THREADS.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/nbody/build}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results}"
NS="${NS:-1024 4096 16384 65536}"
BLOCKS="${BLOCKS:-64 128 256 512}"
BH_NS="${BH_NS:-256 512 1024 2048 4096 8192 16384 32768 65536 131072}"
QUICK="${QUICK:-0}"
mkdir -p "$RESULTS_DIR"

# ── Build ────────────────────────────────────────────────────────────────────
CMAKE_ARGS=(-S "$ROOT/nbody" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release)
[ -n "${CUDA_ARCH:-}" ] && CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH")
cmake "${CMAKE_ARGS[@]}" > /dev/null
cmake --build "$BUILD_DIR" -j"$(nproc)" > /dev/null
CPU_BIN="$BUILD_DIR/nbody_bench_cpu"
GPU_BIN="$BUILD_DIR/nbody_bench_gpu"
SIM_BIN="$BUILD_DIR/galaxy_nbody"

# Steps per run: aim for a roughly constant amount of work per measurement.
#   steps = clamp(budget / N², lo, hi)
steps_for() {  # $1 = N, $2 = budget, $3 = lo, $4 = hi
    local s=$(( $2 / ($1 * $1) ))
    [ "$QUICK" = 1 ] && s=$(( s / 10 ))
    (( s < $3 )) && s=$3
    (( s > $4 )) && s=$4
    echo "$s"
}

# ── System info ──────────────────────────────────────────────────────────────
{
    echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "cpu: $(lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -1)"
    echo "cpu_threads: $(nproc)"
    echo "cxx: $(c++ --version | head -1)"
    if command -v nvidia-smi > /dev/null && nvidia-smi > /dev/null 2>&1; then
        echo "gpu: $(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1)"
        echo "nvcc: $(nvcc --version | tail -1)"
    fi
} > "$RESULTS_DIR/system_$( [ "${SKIP_CPU:-0}" = 1 ] && echo gpu || echo cpu ).txt"

# ── CPU ──────────────────────────────────────────────────────────────────────
if [ "${SKIP_CPU:-0}" != 1 ]; then
    echo "== CPU scaling (serial, OpenMP) =="
    out="$RESULTS_DIR/cpu_scaling.csv"
    "$CPU_BIN" --header > "$out"
    for n in $NS; do
        s=$(steps_for "$n" $((1 << 31)) 2 50)
        for impl in serial omp; do
            "$CPU_BIN" --impl "$impl" --n "$n" --steps "$s" --warmup 1 \
                ${THREADS:+--threads $THREADS} | tee -a "$out"
        done
    done

    echo "== Barnes-Hut vs direct crossover =="
    out="$RESULTS_DIR/bh_crossover.csv"
    "$CPU_BIN" --header > "$out"
    for n in $BH_NS; do
        s=$(steps_for "$n" $((1 << 31)) 1 50)
        for impl in serial omp; do
            "$CPU_BIN" --impl "$impl" --n "$n" --steps "$s" | tee -a "$out"
        done
        sb=$(steps_for "$n" $((1 << 34)) 2 50)
        "$CPU_BIN" --impl bh --n "$n" --steps "$sb" --threads 1 --force-error | tee -a "$out"
        "$CPU_BIN" --impl bh --n "$n" --steps "$sb" --force-error | tee -a "$out"
    done

    echo "== Energy conservation: KDK vs semi-implicit Euler (CPU, N=1024) =="
    out="$RESULTS_DIR/energy_cpu.csv"
    : > "$out"
    first=1
    for integ in kdk euler; do
        for dt in 0.01 0.005 0.0025; do
            steps=$(python3 -c "print(round(20 / $dt))")
            tmp="$(mktemp)"
            "$CPU_BIN" --impl omp --n 1024 --dt "$dt" --steps "$steps" \
                --integrator "$integ" --energy-log "$tmp" \
                --energy-every $(( steps / 100 )) > /dev/null
            if [ $first = 1 ]; then cat "$tmp" >> "$out"; first=0
            else tail -n +2 "$tmp" >> "$out"; fi
            rm -f "$tmp"
            echo "  $integ dt=$dt done"
        done
    done
fi

# ── GPU ──────────────────────────────────────────────────────────────────────
if [ "${SKIP_GPU:-0}" != 1 ] && [ -x "$GPU_BIN" ] && nvidia-smi > /dev/null 2>&1; then
    echo "== GPU scaling (naive vs tiled, block 256) =="
    out="$RESULTS_DIR/gpu_scaling.csv"
    "$GPU_BIN" --header > "$out"
    for n in $NS; do
        s=$(steps_for "$n" $((1 << 32)) 20 500)
        for k in naive tiled; do
            "$GPU_BIN" --kernel "$k" --n "$n" --block 256 --steps "$s" --energy | tee -a "$out"
        done
    done

    echo "== GPU block-size sweep =="
    out="$RESULTS_DIR/gpu_block_sweep.csv"
    "$GPU_BIN" --header > "$out"
    for n in $NS; do
        s=$(steps_for "$n" $((1 << 32)) 20 500)
        for k in naive tiled; do
            for b in $BLOCKS; do
                "$GPU_BIN" --kernel "$k" --n "$n" --block "$b" --steps "$s" | tee -a "$out"
            done
        done
    done

    echo "== Energy conservation on the GPU (galaxy run, N=4096) =="
    for integ in kdk euler; do
        (cd "$RESULTS_DIR" && "$SIM_BIN" 4096 2000 --integrator "$integ" \
            --out /dev/null --energy-log "energy_gpu_$integ.csv" | tail -4)
    done
else
    echo "== GPU benchmarks skipped (no CUDA build or no GPU visible) =="
fi

echo
echo "Results in $RESULTS_DIR.  Next: python3 scripts/make_report.py"
