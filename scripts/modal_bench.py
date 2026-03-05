"""modal_bench.py — Run every benchmark on one Modal GPU container.

    modal run scripts/modal_bench.py              # CPU + GPU + Nsight + report
    modal run scripts/modal_bench.py --quick      # smoke test (fewer steps)
    modal run scripts/modal_bench.py --ncu-only   # redo only the Nsight profile + report

Runs scripts/run_benchmarks.sh (CPU and GPU on the same machine, so "speedup
vs serial" is a same-host comparison), then scripts/profile_ncu.sh, then
scripts/make_report.py, and copies results/ and README.md back here.

The container's `nproc` reports the host's cores, not the cores Modal
reserved for us, so OpenMP is pinned to CPU_CORES and the system file is
corrected to say so.
"""
import os
import pathlib
import subprocess

import modal

GPU = "A100-40GB"
CPU_CORES = 8
ROOT = pathlib.Path(__file__).resolve().parent.parent

app = modal.App("nbody-benchmarks")
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu24.04", add_python="3.12")
    .apt_install("cmake", "build-essential", "cuda-nsight-compute-13-0")
    .uv_pip_install("numpy", "matplotlib")
    .add_local_dir(ROOT / "nbody", remote_path="/repo/nbody", ignore=["build", "*.bin", "*.gif"])
    .add_local_dir(ROOT / "scripts", remote_path="/repo/scripts", ignore=["__pycache__"])
    .add_local_dir(ROOT / "results", remote_path="/repo/results")
    .add_local_file(ROOT / "README.md", remote_path="/repo/README.md")
)


def fix_system_file() -> None:
    """nproc/lscpu describe the host; record what the benchmarks actually used."""
    sysfile = pathlib.Path("/repo/results/system_cpu.txt")
    # (GNU nproc obeys OMP_NUM_THREADS, and lscpu has no model name in the sandbox.)
    host = os.cpu_count()
    model = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "")
    flags = next((l for l in open("/proc/cpuinfo") if l.startswith("flags")), "")
    simd = "AVX-512" if " avx512f" in flags else "AVX2" if " avx2" in flags else "SSE"
    lines = [f"cpu_threads: {CPU_CORES}" if l.startswith("cpu_threads:")
             else f"cpu: {model or 'x86-64 (model name hidden by the Modal sandbox)'} [{simd}]" if l.startswith("cpu:") and "unknown" in l
             else l for l in sysfile.read_text().splitlines()]
    lines.append(f"note: Modal container, {CPU_CORES} reserved cores (OMP_NUM_THREADS={CPU_CORES}); host reports {host}")
    sysfile.write_text("\n".join(lines) + "\n")


@app.function(image=image, gpu=GPU, cpu=CPU_CORES, memory=16384, timeout=2 * 60 * 60)
def run(quick: bool = False, ncu_only: bool = False) -> dict:
    # ncu cannot lock GPU clocks inside the sandbox; profile without fixed frequencies.
    env = dict(os.environ, OMP_NUM_THREADS=str(CPU_CORES), QUICK="1" if quick else "0",
               OMP_PROC_BIND="true", NCU_ARGS="--clock-control none")
    notes = []

    def sh(cmd: str, check: bool = True) -> int:
        print(f"\n$ {cmd}", flush=True)
        p = subprocess.run(cmd, shell=True, cwd="/repo", env=env)
        if check and p.returncode != 0:
            raise RuntimeError(f"`{cmd}` exited {p.returncode}")
        return p.returncode

    sh("nvidia-smi && nproc && lscpu | head -20")
    if ncu_only:
        sh("cmake -S nbody -B nbody/build -DCMAKE_BUILD_TYPE=Release > /dev/null && "
           "cmake --build nbody/build -j8 > /dev/null")
    else:
        sh("scripts/run_benchmarks.sh")
        fix_system_file()
    # Nsight Compute needs GPU performance counters, which a sandbox may refuse.
    ncu = subprocess.run("ls /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1", shell=True,
                         capture_output=True, text=True).stdout.strip()
    if ncu:
        env["PATH"] = os.path.dirname(ncu) + ":" + env["PATH"]
    if sh("scripts/profile_ncu.sh 2>&1 | tee /tmp/ncu.log", check=False) != 0 \
            or not os.path.exists("/repo/results/ncu_metrics.csv"):
        diag = subprocess.run(
            "ncu --version; ncu --kernel-name regex:accel_naive_kernel --launch-count 1 "
            "nbody/build/nbody_bench_gpu --kernel naive --n 4096 --block 256 --steps 2 2>&1 | tail -25",
            shell=True, cwd="/repo", env=env, capture_output=True, text=True)
        notes.append("ncu failed: " + open("/tmp/ncu.log").read()[-800:] + "\n--- direct run ---\n"
                     + diag.stdout[-2500:] + diag.stderr[-800:])

    sh("python3 scripts/make_report.py")

    files = {str(p.relative_to("/repo")): p.read_bytes()
             for p in pathlib.Path("/repo/results").iterdir() if p.is_file()}
    files["README.md"] = pathlib.Path("/repo/README.md").read_bytes()
    return {"files": files, "notes": notes}


@app.local_entrypoint()
def main(quick: bool = False, ncu_only: bool = False):
    out = run.remote(quick, ncu_only)
    for rel, data in out["files"].items():
        path = ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print("wrote", rel)
    for note in out["notes"]:
        print("\nNOTE:", note)
