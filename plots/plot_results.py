#!/usr/bin/env python3
"""plot_results.py - turns the CSV files in results/ into charts and a table.

    python3 plots/plot_results.py        (needs: pip install matplotlib)

Reads   results/benchmarks.csv, results/block_sweep.csv, results/energy.csv
Writes  results/scaling.png      time per step vs number of bodies
        results/block_sweep.png  GPU speed vs block size
        results/energy.png       energy error over time, KDK vs Euler
        results/table.md         the benchmark numbers as a Markdown table
Files that are missing are skipped.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")   # draw to files, no window needed
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

# One fixed name, colour and marker per version, used in every chart.
NAMES   = {"cpu_serial": "CPU, 1 thread", "cpu_openmp": "CPU, OpenMP",
           "gpu_naive": "GPU, naive", "gpu_tiled": "GPU, tiled"}
COLORS  = {"cpu_serial": "#2a78d6", "cpu_openmp": "#eb6834",
           "gpu_naive": "#1baf7a", "gpu_tiled": "#6250d6",
           "kdk": "#2a78d6", "euler": "#eb6834"}
MARKERS = {"cpu_serial": "o", "cpu_openmp": "s", "gpu_naive": "^", "gpu_tiled": "D"}

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.grid": True, "grid.color": "#e4e3df", "axes.edgecolor": "#e4e3df",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "lines.linewidth": 2, "lines.markersize": 7, "legend.frameon": False,
})


def read_csv(name):
    """Returns the rows of results/<name> as a list of dicts ([] if missing)."""
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def short(n):
    """1024 -> '1K', 65536 -> '64K'"""
    return f"{n // 1024}K"


def plot_scaling(rows):
    """Time per step against N, one line per version."""
    sizes = sorted({int(r["n"]) for r in rows})
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for impl in NAMES:
        mine = sorted((int(r["n"]), float(r["ms_per_step"])) for r in rows if r["impl"] == impl)
        if mine:
            ax.plot([n for n, _ in mine], [ms for _, ms in mine], label=NAMES[impl],
                    color=COLORS[impl], marker=MARKERS[impl], markeredgecolor="white")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([short(n) for n in sizes])
    ax.set_xlabel("number of bodies N")
    ax.set_ylabel("milliseconds per step")
    ax.set_title("Time per step (lower is better)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "scaling.png"), dpi=150)
    plt.close(fig)


def plot_block_sweep(rows):
    """GFLOP/s against block size, one small chart per N."""
    sizes  = sorted({int(r["n"]) for r in rows})
    blocks = sorted({int(r["block"]) for r in rows})
    fig, axes = plt.subplots(1, len(sizes), figsize=(3.2 * len(sizes), 3.4))
    for ax, n in zip(axes, sizes):
        for impl in ("gpu_naive", "gpu_tiled"):
            mine = sorted((int(r["block"]), float(r["gflops"])) for r in rows
                          if r["impl"] == impl and int(r["n"]) == n)
            ax.plot([b for b, _ in mine], [g for _, g in mine], label=NAMES[impl],
                    color=COLORS[impl], marker=MARKERS[impl], markeredgecolor="white")
        ax.set_xscale("log", base=2)
        ax.set_xticks(blocks)
        ax.set_xticklabels(blocks)
        ax.set_ylim(bottom=0)
        ax.set_title(f"N = {short(n)}")
        ax.set_xlabel("threads per block")
    axes[0].set_ylabel("GFLOP/s (higher is better)")
    axes[-1].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "block_sweep.png"), dpi=150)
    plt.close(fig)


def plot_energy(rows):
    """Size of the energy error over time, one small chart per time step."""
    time_steps = sorted({float(r["dt"]) for r in rows}, reverse=True)
    fig, axes = plt.subplots(1, len(time_steps), figsize=(4 * len(time_steps), 3.6), sharey=True)
    for ax, dt in zip(axes, time_steps):
        for integrator, label in (("kdk", "KDK leapfrog"), ("euler", "Euler")):
            mine = [r for r in rows if r["integrator"] == integrator and float(r["dt"]) == dt]
            ax.plot([float(r["time"]) for r in mine],
                    [abs(float(r["energy_error"])) for r in mine],
                    label=label, color=COLORS[integrator])
        ax.set_yscale("log")
        ax.set_title(f"dt = {dt:g}")
        ax.set_xlabel("simulation time")
    axes[0].set_ylabel("energy error  |E - E0| / |E0|")
    axes[-1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "energy.png"), dpi=150)
    plt.close(fig)


def write_table(rows):
    """Markdown table: one row per N and version, with speedup vs 1 CPU thread."""
    lines = ["| N | version | ms per step | speedup | interactions / s | GFLOP/s |",
             "|---|---|--:|--:|--:|--:|"]
    for n in sorted({int(r["n"]) for r in rows}):
        mine   = {r["impl"]: r for r in rows if int(r["n"]) == n}
        serial = float(mine["cpu_serial"]["ms_per_step"]) if "cpu_serial" in mine else None
        for impl in NAMES:
            if impl not in mine:
                continue
            ms      = float(mine[impl]["ms_per_step"])
            speedup = f"{serial / ms:.1f}x" if serial else "-"
            billions = float(mine[impl]["interactions_per_s"]) / 1e9
            lines.append(f"| {short(n)} | {NAMES[impl]} | {ms:.3f} | {speedup} | "
                         f"{billions:.1f} billion | {float(mine[impl]['gflops']):,.0f} |")
    with open(os.path.join(RESULTS, "table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


benchmarks  = read_csv("benchmarks.csv")
block_sweep = read_csv("block_sweep.csv")
energy      = read_csv("energy.csv")

if benchmarks:
    plot_scaling(benchmarks)
    write_table(benchmarks)
if block_sweep:
    plot_block_sweep(block_sweep)
if energy:
    plot_energy(energy)
print("Charts and table written to results/")
