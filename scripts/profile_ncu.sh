#!/usr/bin/env bash
# profile_ncu.sh  —  Nsight Compute profile of the naive vs tiled force kernels.
#
#   scripts/profile_ncu.sh                 # N = 16384 and 65536, block 256
#   NS="65536" BLOCK=128 scripts/profile_ncu.sh
#   FULL=1 scripts/profile_ncu.sh          # also save full .ncu-rep reports
#                                          # (open with ncu-ui)
#
# Writes results/ncu_metrics.csv (one row per kernel × N) for make_report.py.
# Needs GPU performance-counter access: run as root, or set
#   NVreg_RestrictProfilingToAdminUsers=0 for the nvidia kernel module.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/nbody/build}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results}"
NS="${NS:-16384 65536}"
BLOCK="${BLOCK:-256}"
BIN="$BUILD_DIR/nbody_bench_gpu"
mkdir -p "$RESULTS_DIR"
[ -x "$BIN" ] || { echo "build first: scripts/run_benchmarks.sh (or cmake)"; exit 1; }
command -v ncu > /dev/null || { echo "ncu (Nsight Compute CLI) not found in PATH"; exit 1; }

# Metric                                                     What it tells you
METRICS=(
  gpu__time_duration.sum                                   # kernel time
  sm__throughput.avg.pct_of_peak_sustained_elapsed         # compute "speed of light" %
  gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed  # memory SOL %
  dram__throughput.avg.pct_of_peak_sustained_elapsed       # DRAM bandwidth %
  dram__bytes_read.sum                                     # bytes from DRAM
  lts__t_bytes.sum                                         # L2 traffic
  l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum             # global loads through L1
  l1tex__data_pipe_lsu_wavefronts_mem_shared.sum           # shared-memory wavefronts
  l1tex__t_sector_hit_rate.pct                             # L1 hit rate
  sm__warps_active.avg.pct_of_peak_sustained_active        # achieved occupancy
  sm__maximum_warps_per_active_cycle_pct                   # theoretical occupancy
  launch__registers_per_thread
  launch__occupancy_limit_registers
  smsp__average_warp_latency_issue_stalled_long_scoreboard # waiting on L1/global
  smsp__average_warp_latency_issue_stalled_barrier         # waiting on __syncthreads
)
METRIC_LIST=$(IFS=,; echo "${METRICS[*]}")

out="$RESULTS_DIR/ncu_metrics.csv"
rm -f "$out"
for n in $NS; do
    for k in naive tiled; do
        echo "== ncu: $k kernel, N=$n, block $BLOCK =="
        log="$(mktemp)"
        # Skip the first force launches (initial accel + warm-up), profile one.
        ncu --kernel-name "regex:accel_${k}_kernel" --launch-skip 3 --launch-count 1 \
            --metrics "$METRIC_LIST" --csv --page raw --log-file "$log" \
            "$BIN" --kernel "$k" --n "$n" --block "$BLOCK" --steps 2 --warmup 2 > /dev/null
        # ncu raw CSV: header row, units row, one data row.  Prepend our labels.
        python3 - "$log" "$out" "$k" "$n" "$BLOCK" <<'EOF'
import csv, os, sys
log, out, kernel, n, block = sys.argv[1:]
rows = [r for r in csv.reader(open(log)) if r and not r[0].startswith("==")]
header, units, data = rows[0], rows[1], rows[2]
new = not os.path.exists(out)
with open(out, "a", newline="") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["impl", "n", "block"] + header)
        w.writerow(["", "", ""] + units)
    w.writerow(["gpu_" + kernel, n, block] + data)
EOF
        rm -f "$log"
        if [ "${FULL:-0}" = 1 ]; then
            ncu --kernel-name "regex:accel_${k}_kernel" --launch-skip 3 --launch-count 1 \
                --set full -f -o "$RESULTS_DIR/ncu_${k}_n${n}" \
                "$BIN" --kernel "$k" --n "$n" --block "$BLOCK" --steps 2 --warmup 2 > /dev/null
        fi
    done
done
echo "Wrote $out.  Next: python3 scripts/make_report.py"
