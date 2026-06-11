#!/usr/bin/env bash
# profile_ncu.sh - profiles the naive and tiled force kernels with
# NVIDIA Nsight Compute (ncu) so we can see WHY one is faster.
#
#   bash bench/profile_ncu.sh
#
# Writes one CSV per run: results/ncu_<kernel>_n<N>.csv
# Each file has a header row, a units row and one row of numbers
# (plus a few "==PROF==" message lines from ncu itself).
#
# ncu needs permission to read the GPU's performance counters, so you may have
# to run this as root. In a cloud container where ncu says it cannot lock the
# GPU clocks, run:  NCU_ARGS="--clock-control none" bash bench/profile_ncu.sh
set -e
cd "$(dirname "$0")/.."
mkdir -p results
make gpu

# The numbers we care about:
METRICS="gpu__time_duration.sum"                                    # how long the kernel ran
METRICS+=",sm__throughput.avg.pct_of_peak_sustained_elapsed"        # how busy the compute units were (%)
METRICS+=",gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"  # how busy memory was (%)
METRICS+=",l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum"            # bytes loaded from global memory
METRICS+=",l1tex__data_pipe_lsu_wavefronts_mem_shared.sum"          # shared memory accesses
METRICS+=",lts__t_bytes.sum"                                        # bytes through the L2 cache
METRICS+=",sm__warps_active.avg.pct_of_peak_sustained_active"       # occupancy actually achieved (%)
METRICS+=",sm__maximum_warps_per_active_cycle_pct"                  # occupancy possible in theory (%)
METRICS+=",launch__registers_per_thread"

for n in 16384 65536; do
    for kernel in naive tiled; do
        echo "profiling the $kernel kernel, N = $n"
        # bench_gpu launches the force kernel many times. Skip the first 4
        # launches (set-up and warm-up) and profile exactly one.
        ncu ${NCU_ARGS:-} --kernel-name "accel_${kernel}_kernel" \
            --launch-skip 4 --launch-count 1 \
            --metrics "$METRICS" --csv --page raw --log-file "results/ncu_${kernel}_n${n}.csv" \
            ./build/bench_gpu $kernel $n 256 2 > /dev/null
    done
done
echo "Done. Results are in results/ncu_*.csv"
