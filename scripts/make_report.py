#!/usr/bin/env python3
"""make_report.py — Turn results/*.csv into charts and the README results section.

    python3 scripts/make_report.py            # needs numpy + matplotlib

Reads whatever CSVs exist in results/ (CPU-only, GPU-only or both), writes
results/*.png, and replaces the text between

    <!-- RESULTS:BEGIN -->  ...  <!-- RESULTS:END -->

in README.md.  Rows that have not been measured yet are shown as "—".
"""
import csv
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
README = os.path.join(ROOT, "README.md")

# Fixed entity → colour mapping (categorical slots, never re-ordered), so a
# solver keeps its colour in every chart even when other series are missing.
COLORS = {
    "cpu_serial":     "#2a78d6",  # slot 1 blue
    "cpu_openmp":     "#eb6834",  # slot 2 orange
    "gpu_naive":      "#1baf7a",  # slot 3 aqua
    "gpu_tiled":      "#6250d6",  # slot 7 violet
    "bh_1thread":     "#2a78d6",  # BH reuses its direct twin's hue, dashed
    "bh_openmp":      "#eb6834",
    "kdk":            "#2a78d6",
    "euler":          "#eb6834",
}
MARKERS = {"cpu_serial": "o", "cpu_openmp": "s", "gpu_naive": "^", "gpu_tiled": "D",
           "bh_1thread": "o", "bh_openmp": "s"}
LABELS = {"cpu_serial": "CPU serial (1 thread, SIMD)",
          "cpu_openmp": "CPU OpenMP",
          "gpu_naive": "GPU naive",
          "gpu_tiled": "GPU tiled (shared mem)",
          "bh_1thread": "Barnes–Hut, 1 thread",
          "bh_openmp": "Barnes–Hut, OpenMP"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
NS = [1024, 4096, 16384, 65536]


# ─── Data loading ────────────────────────────────────────────────────────────
def read_csv(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def num(x):
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def read_text(name):
    path = os.path.join(RES, name)
    return open(path).read().strip() if os.path.exists(path) else ""


cpu = read_csv("cpu_scaling.csv")
gpu = read_csv("gpu_scaling.csv")
sweep = read_csv("gpu_block_sweep.csv")
bh = read_csv("bh_crossover.csv")
energy_cpu = read_csv("energy_cpu.csv")
ncu = [r for r in read_csv("ncu_metrics.csv") if r.get("impl")]

scaling = {}  # (impl, n) -> row
for r in cpu + gpu:
    scaling[(r["impl"], int(r["n"]))] = r


def bh_key(r):
    if r["impl"] == "cpu_barnes_hut":
        return "bh_1thread" if r["threads"] == "1" else "bh_openmp"
    return r["impl"]


bh_rows = {(bh_key(r), int(r["n"])): r for r in bh}


# ─── Formatting helpers ──────────────────────────────────────────────────────
def fmt_ms(v):
    if v is None:
        return "—"
    return f"{v:,.3f}" if v < 10 else f"{v:,.1f}"


def fmt_si(v, unit=""):
    if v is None:
        return "—"
    for p, s in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(v) >= p:
            return f"{v / p:.2f} {s}{unit}"
    return f"{v:.2f} {unit}"


def fmt_n(n):
    return f"{n // 1024}K" if n >= 1024 and n % 1024 == 0 else str(n)


# ─── Charts ──────────────────────────────────────────────────────────────────
def setup_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping charts", file=sys.stderr)
        return None
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "legend.frameon": False,
        "lines.linewidth": 2, "lines.markersize": 7,
    })
    return plt


def log2_xaxis(ax, ns):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([fmt_n(n) for n in ns])
    ax.minorticks_off()


def chart_scaling(plt):
    impls = [i for i in ("cpu_serial", "cpu_openmp", "gpu_naive", "gpu_tiled")
             if any((i, n) in scaling for n in NS)]
    if not impls:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for impl in impls:
        ns = [n for n in NS if (impl, n) in scaling]
        ms = [num(scaling[(impl, n)]["ms_per_step"]) for n in ns]
        ips = [num(scaling[(impl, n)]["interactions_per_s"]) for n in ns]
        kw = dict(color=COLORS[impl], marker=MARKERS[impl], label=LABELS[impl],
                  markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax1.plot(ns, ms, **kw)
        ax2.plot(ns, ips, **kw)
    for ax in (ax1, ax2):
        log2_xaxis(ax, NS)
        ax.set_yscale("log")
        ax.set_xlabel("bodies N")
    ax1.set_title("Time per step (lower is better)")
    ax1.set_ylabel("ms / step")
    ax2.set_title("Throughput (higher is better)")
    ax2.set_ylabel("body–body interactions / s")
    ax2.legend(loc="lower right")
    fig.tight_layout()
    return save(fig, plt, "scaling.png")


def chart_bh(plt):
    series = ["cpu_serial", "bh_1thread", "cpu_openmp", "bh_openmp"]
    ns_all = sorted({n for (_, n) in bh_rows})
    if not ns_all:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for s in series:
        ns = [n for n in ns_all if (s, n) in bh_rows]
        if not ns:
            continue
        ms = [num(bh_rows[(s, n)]["ms_per_step"]) for n in ns]
        ax1.plot(ns, ms, color=COLORS[s], marker=MARKERS[s], label=LABELS[s],
                 linestyle="--" if s.startswith("bh") else "-",
                 markeredgecolor=SURFACE, markeredgewidth=1.5)
    log2_xaxis(ax1, ns_all[::2] if len(ns_all) > 6 else ns_all)
    ax1.set_yscale("log")
    ax1.set_xlabel("bodies N")
    ax1.set_ylabel("ms / step")
    ax1.set_title("Barnes–Hut (θ = 0.5) vs direct summation")
    ax1.legend(loc="upper left")
    for s in ("bh_1thread", "bh_openmp"):
        ns = [n for n in ns_all if (s, n) in bh_rows
              and num(bh_rows[(s, n)]["force_rel_err"]) is not None]
        if s == "bh_openmp" and ns:   # same tree → same error; plot once
            err = [num(bh_rows[(s, n)]["force_rel_err"]) * 100 for n in ns]
            ax2.plot(ns, err, color=INK2, marker="o", markeredgecolor=SURFACE,
                     markeredgewidth=1.5)
            log2_xaxis(ax2, ns[::2] if len(ns) > 6 else ns)
    ax2.set_xlabel("bodies N")
    ax2.set_ylabel("RMS relative force error (%)")
    ax2.set_title("Barnes–Hut accuracy (vs exact direct sum)")
    ax2.set_ylim(bottom=0)
    fig.tight_layout()
    return save(fig, plt, "bh_crossover.png")


def chart_energy(plt):
    if not energy_cpu:
        return None
    dts = sorted({float(r["dt"]) for r in energy_cpu}, reverse=True)
    fig, axes = plt.subplots(1, len(dts), figsize=(4 * len(dts), 3.6), sharey=True)
    axes = axes if len(dts) > 1 else [axes]
    for ax, dt in zip(axes, dts):
        for integ in ("kdk", "euler"):
            rows = [r for r in energy_cpu if r["integrator"] == integ
                    and float(r["dt"]) == dt and float(r["time"]) > 0]  # ΔE(0) = 0
            t = [float(r["time"]) for r in rows]
            d = [max(abs(float(r["rel_drift"])), 1e-12) for r in rows]
            ax.plot(t, d, color=COLORS[integ],
                    label="KDK leapfrog" if integ == "kdk" else "semi-implicit Euler")
        ax.set_yscale("log")
        ax.set_title(f"dt = {dt:g}")
        ax.set_xlabel("simulation time")
    axes[0].set_ylabel("|E(t) − E₀| / |E₀|")
    axes[-1].legend(loc="lower right")
    fig.suptitle("Energy drift, N = 1024 galaxy (CPU, same physics as the GPU path)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return save(fig, plt, "energy.png")


def chart_block_sweep(plt):
    if not sweep:
        return None
    blocks = sorted({int(r["block"]) for r in sweep})
    ns = sorted({int(r["n"]) for r in sweep})
    fig, axes = plt.subplots(1, len(ns), figsize=(3.2 * len(ns), 3.4), sharey=False)
    axes = axes if len(ns) > 1 else [axes]
    for ax, n in zip(axes, ns):
        for impl in ("gpu_naive", "gpu_tiled"):
            rows = {int(r["block"]): r for r in sweep if r["impl"] == impl and int(r["n"]) == n}
            bs = [b for b in blocks if b in rows]
            ax.plot(bs, [num(rows[b]["gflops"]) for b in bs], color=COLORS[impl],
                    marker=MARKERS[impl], label=LABELS[impl],
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
        log2_xaxis(ax, blocks)
        ax.set_xticklabels([str(b) for b in blocks])
        ax.set_title(f"N = {fmt_n(n)}")
        ax.set_xlabel("threads / block")
    axes[0].set_ylabel("GFLOP/s (20 FLOP/interaction)")
    axes[-1].legend(loc="lower right")
    fig.tight_layout()
    return save(fig, plt, "block_sweep.png")


def save(fig, plt, name):
    path = os.path.join(RES, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return name


# ─── Markdown tables ─────────────────────────────────────────────────────────
def table(header, rows, align=None):
    align = align or ["---"] + ["--:"] * (len(header) - 1)
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def scaling_table():
    rows = []
    for n in NS:
        base = num(scaling.get(("cpu_serial", n), {}).get("ms_per_step"))
        for impl in ("cpu_serial", "cpu_openmp", "gpu_naive", "gpu_tiled"):
            r = scaling.get((impl, n), {})
            ms = num(r.get("ms_per_step"))
            sp = f"{base / ms:,.1f}×" if (ms and base) else "—"
            drift = num(r.get("energy_drift"))
            rows.append([f"{fmt_n(n)}" if impl == "cpu_serial" else "",
                         LABELS[impl], fmt_ms(ms), sp,
                         fmt_si(num(r.get("interactions_per_s")), "/s"),
                         f"{num(r['gflops']):,.1f}" if ms else "—",
                         f"{drift:.1e}" if drift is not None else "—"])
    return table(["N", "implementation", "ms / step", "speedup vs serial",
                  "interactions / s", "GFLOP/s", "ΔE/E₀ (run)"],
                 rows, ["---", "---", "--:", "--:", "--:", "--:", "--:"])


def sweep_table():
    if not sweep:
        return "_Not measured yet — run `scripts/run_benchmarks.sh` on a machine with an NVIDIA GPU._"
    blocks = sorted({int(r["block"]) for r in sweep})
    rows = []
    for n in sorted({int(r["n"]) for r in sweep}):
        for impl in ("gpu_naive", "gpu_tiled"):
            d = {int(r["block"]): num(r["gflops"]) for r in sweep
                 if r["impl"] == impl and int(r["n"]) == n}
            best = max((v for v in d.values() if v), default=None)
            cells = [(f"**{d[b]:,.0f}**" if d.get(b) == best else f"{d[b]:,.0f}")
                     if d.get(b) else "—" for b in blocks]
            rows.append([fmt_n(n), LABELS[impl]] + cells)
    return table(["N", "kernel"] + [f"block {b}" for b in blocks], rows,
                 ["---", "---"] + ["--:"] * len(blocks)) + \
        "\n\nGFLOP/s; best block size per row in bold."


def crossover(a, b):
    """Smallest N (log-interpolated) where series b becomes faster than a."""
    ns = sorted(n for (s, n) in bh_rows if s == a and (b, n) in bh_rows)
    prev = None
    for n in ns:
        ratio = num(bh_rows[(a, n)]["ms_per_step"]) / num(bh_rows[(b, n)]["ms_per_step"])
        if ratio > 1:
            if prev is None:
                return f"≤ {fmt_n(n)}"
            (n0, r0) = prev
            f = (0 - math.log(r0)) / (math.log(ratio) - math.log(r0))
            nx = 2 ** (math.log2(n0) + f * (math.log2(n) - math.log2(n0)))
            return f"≈ {nx:,.0f}"
        prev = (n, ratio)
    return "not reached"


def bh_table():
    if not bh_rows:
        return "_Not measured yet._"
    rows = []
    for n in sorted({n for (_, n) in bh_rows}):
        g = lambda s: num(bh_rows.get((s, n), {}).get("ms_per_step"))
        err = num(bh_rows.get(("bh_openmp", n), {}).get("force_rel_err"))
        rows.append([fmt_n(n), fmt_ms(g("cpu_serial")), fmt_ms(g("bh_1thread")),
                     fmt_ms(g("cpu_openmp")), fmt_ms(g("bh_openmp")),
                     f"{err * 100:.2f} %" if err is not None else "—"])
    return table(["N", "direct, 1 thread", "Barnes–Hut, 1 thread",
                  "direct, OpenMP", "Barnes–Hut, OpenMP", "BH force error (RMS)"], rows) + \
        f"\n\nms per step. **Crossover** (Barnes–Hut faster than direct summation on the " \
        f"same threads): 1 thread **N {crossover('cpu_serial', 'bh_1thread')}**, OpenMP " \
        f"**N {crossover('cpu_openmp', 'bh_openmp')}**."


def energy_table():
    if not energy_cpu:
        return "_Not measured yet._"
    rows = []
    for integ in ("kdk", "euler"):
        for dt in sorted({float(r["dt"]) for r in energy_cpu}, reverse=True):
            rs = [r for r in energy_cpu if r["integrator"] == integ and float(r["dt"]) == dt]
            if not rs:
                continue
            final = float(rs[-1]["rel_drift"])
            mx = max(abs(float(r["rel_drift"])) for r in rs)
            rows.append(["KDK leapfrog" if integ == "kdk" else "semi-implicit Euler",
                         f"{dt:g}", rs[-1]["step"], f"{final:+.2e}", f"{mx:.2e}"])
    return table(["integrator", "dt", "steps", "final ΔE/E₀", "max abs(ΔE/E₀)"], rows,
                 ["---", "--:", "--:", "--:", "--:"])


NCU_COLS = [
    ("gpu__time_duration.sum", "kernel time"),
    ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "compute throughput (% peak)"),
    ("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", "memory throughput (% peak)"),
    ("dram__throughput.avg.pct_of_peak_sustained_elapsed", "DRAM throughput (% peak)"),
    ("lts__t_bytes.sum", "L2 traffic"),
    ("l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum", "global-load bytes (L1)"),
    ("l1tex__data_pipe_lsu_wavefronts_mem_shared.sum", "shared-mem wavefronts"),
    ("sm__warps_active.avg.pct_of_peak_sustained_active", "achieved occupancy (%)"),
    ("sm__maximum_warps_per_active_cycle_pct", "theoretical occupancy (%)"),
    ("launch__registers_per_thread", "registers / thread"),
    ("smsp__average_warp_latency_issue_stalled_long_scoreboard", "stall: long scoreboard (cyc)"),
    ("smsp__average_warp_latency_issue_stalled_barrier", "stall: barrier (cyc)"),
]


def ncu_table():
    if not ncu:
        return ("_Not measured yet — run `scripts/profile_ncu.sh` on a machine with an "
                "NVIDIA GPU and Nsight Compute._")
    units = {}
    raw = read_csv("ncu_metrics.csv")
    if raw and not raw[0].get("impl"):
        units = raw[0]
    cols = sorted({(int(r["n"]), r["impl"]) for r in ncu})
    header = ["metric"] + [f"{LABELS[i]} N={fmt_n(n)}" for (n, i) in cols]
    rows = []
    for key, label in NCU_COLS:
        u = units.get(key, "")
        row = [f"{label}" + (f" [{u}]" if u and u != "%" else "")]
        for (n, i) in cols:
            r = next(x for x in ncu if int(x["n"]) == n and x["impl"] == i)
            v = r.get(key, "")
            row.append(v.replace(",", "") if v else "—")
        rows.append(row)
    return table(header, rows, ["---"] + ["--:"] * len(cols))


# ─── README section ──────────────────────────────────────────────────────────
def build_section(charts):
    sys_cpu = read_text("system_cpu.txt")
    sys_gpu = read_text("system_gpu.txt") or ""
    gpu_line = next((l for l in (sys_gpu + "\n" + sys_cpu).splitlines() if l.startswith("gpu:")), "")
    cpu_line = next((l for l in sys_cpu.splitlines() if l.startswith("cpu:")), "")
    thr_line = next((l for l in sys_cpu.splitlines() if l.startswith("cpu_threads:")), "")
    parts = [
        "_Generated by `scripts/make_report.py` from `results/*.csv` — do not edit by hand._",
        "",
        "**Hardware:** " + ", ".join(x for x in [
            cpu_line.replace("cpu: ", "CPU ") + (f" ({thr_line.split(': ')[1]} threads)" if thr_line else ""),
            gpu_line.replace("gpu: ", "GPU ") if gpu_line else "GPU: _not measured yet_"] if x),
        "",
        "### Direct summation: time per step, speedup, throughput",
        "",
        "One step = one force evaluation (N² interactions) + one KDK integration kernel. "
        "GFLOP/s uses the 20 FLOP/interaction convention. Speedup is against the "
        "single-thread CPU code at the same N.",
        "",
        scaling_table(),
        "",
    ]
    if "scaling.png" in charts:
        parts += ["![Time per step and throughput vs N](results/scaling.png)", ""]
    parts += ["### GPU block-size sweep", "", sweep_table(), ""]
    if "block_sweep.png" in charts:
        parts += ["![GFLOP/s vs block size](results/block_sweep.png)", ""]
    parts += ["### Nsight Compute: naive vs tiled", "", ncu_table(), ""]
    parts += ["### Energy conservation", "", energy_table(), ""]
    if "energy.png" in charts:
        parts += ["![Energy drift vs time](results/energy.png)", ""]
    parts += ["### Barnes–Hut vs direct summation (CPU)", "", bh_table(), ""]
    if "bh_crossover.png" in charts:
        parts += ["![Barnes–Hut crossover](results/bh_crossover.png)", ""]
    return "\n".join(parts)


def main():
    plt = setup_mpl()
    charts = []
    if plt:
        for fn in (chart_scaling, chart_block_sweep, chart_energy, chart_bh):
            name = fn(plt)
            if name:
                charts.append(name)
    section = build_section(charts)
    with open(os.path.join(RES, "RESULTS.md"), "w") as f:
        f.write("# Benchmark results\n\n" + section.replace("](results/", "](") + "\n")
    text = open(README).read()
    pat = re.compile(r"(<!-- RESULTS:BEGIN -->\n).*?(<!-- RESULTS:END -->)", re.S)
    if pat.search(text):
        text = pat.sub(lambda m: m.group(1) + section + "\n" + m.group(2), text)
        open(README, "w").write(text)
        print("Updated README.md results section")
    else:
        print("README.md has no RESULTS markers; wrote results/RESULTS.md only")
    print("Charts:", ", ".join(charts) or "none")


if __name__ == "__main__":
    main()
