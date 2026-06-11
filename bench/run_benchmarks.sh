#!/usr/bin/env bash
# run_benchmarks.sh - runs every benchmark and saves the numbers in results/.
#
#   bash bench/run_benchmarks.sh
#
# Writes:
#   results/benchmarks.csv   time per step for each version at N = 1K..64K
#   results/block_sweep.csv  GPU kernels at block sizes 64, 128, 256, 512
#   results/energy.csv       energy error over time, KDK vs Euler
#   results/machine.txt      what hardware this ran on
# The GPU parts are skipped if there is no NVIDIA GPU.
# Afterwards run:  python3 plots/plot_results.py
set -e
cd "$(dirname "$0")/.."
mkdir -p results

HEADER="impl,n,block,ms_per_step,interactions_per_s,gflops"

# Bigger N takes much longer per step, so use fewer steps.
steps_for() {
    case $1 in
        1024)  echo 200 ;;
        4096)  echo 50  ;;
        16384) echo 10  ;;
        65536) echo 3   ;;
    esac
}

have_gpu=no
if command -v nvidia-smi > /dev/null && nvidia-smi > /dev/null 2>&1; then
    have_gpu=yes
fi

# ---- build ----------------------------------------------------------------
make cpu
if [ $have_gpu = yes ]; then
    make gpu
    ./build/check_gpu    # stop here if the GPU does not agree with the CPU
fi

# ---- what machine is this? --------------------------------------------------
{
    echo "date: $(date -u +%Y-%m-%d)"
    echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"
    echo "cpu_threads: ${OMP_NUM_THREADS:-$(nproc)}"
    if [ $have_gpu = yes ]; then
        echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    fi
} > results/machine.txt

# ---- 1. time per step: CPU serial, CPU OpenMP, GPU naive, GPU tiled --------
echo "$HEADER" > results/benchmarks.csv
for n in 1024 4096 16384 65536; do
    steps=$(steps_for $n)
    ./build/bench_cpu serial $n $steps | tee -a results/benchmarks.csv
    ./build/bench_cpu omp    $n $steps | tee -a results/benchmarks.csv
    if [ $have_gpu = yes ]; then
        # The GPU is fast, so it can afford 10x more steps for a steadier average.
        ./build/bench_gpu naive $n 256 $((steps * 10)) | tee -a results/benchmarks.csv
        ./build/bench_gpu tiled $n 256 $((steps * 10)) | tee -a results/benchmarks.csv
    fi
done

# ---- 2. GPU block size sweep -------------------------------------------------
if [ $have_gpu = yes ]; then
    echo "$HEADER" > results/block_sweep.csv
    for n in 1024 4096 16384 65536; do
        steps=$(( $(steps_for $n) * 10 ))
        for kernel in naive tiled; do
            for block in 64 128 256 512; do
                ./build/bench_gpu $kernel $n $block $steps | tee -a results/block_sweep.csv
            done
        done
    done
fi

# ---- 3. energy error: KDK vs Euler at three time steps ----------------------
echo "integrator,dt,time,energy_error" > results/energy.csv
for integrator in kdk euler; do
    for dt in 0.01 0.005 0.0025; do
        echo "energy test: $integrator dt=$dt"
        ./build/energy_drift $integrator $dt >> results/energy.csv
    done
done

echo "Done. Now run: python3 plots/plot_results.py"
