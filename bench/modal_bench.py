"""modal_bench.py - runs all the benchmarks on a rented cloud GPU with Modal.

    modal run bench/modal_bench.py

It copies the code into a container that has an NVIDIA A100, runs
bench/run_benchmarks.sh, bench/profile_ncu.sh and plots/plot_results.py there,
and copies the results/ folder back to this machine.

The CPU and GPU numbers come from the same machine, so the speedups compare
like with like.
"""
import os
import pathlib
import subprocess

import modal

GPU = "A100-40GB"
CPU_CORES = 8
ROOT = pathlib.Path(__file__).resolve().parent.parent

app = modal.App("nbody-benchmarks")

# The container: NVIDIA's CUDA image plus make, Nsight Compute and matplotlib,
# with this project's folders copied in under /repo.
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu24.04", add_python="3.12")
    .apt_install("build-essential", "cuda-nsight-compute-13-0")
    .uv_pip_install("numpy", "matplotlib")
    .add_local_dir(ROOT / "src", remote_path="/repo/src")
    .add_local_dir(ROOT / "bench", remote_path="/repo/bench", ignore=["__pycache__"])
    .add_local_dir(ROOT / "plots", remote_path="/repo/plots", ignore=["__pycache__"])
    .add_local_file(ROOT / "Makefile", remote_path="/repo/Makefile")
)


@app.function(image=image, gpu=GPU, cpu=CPU_CORES, memory=16384, timeout=2 * 60 * 60)
def run() -> dict:
    # Inside the container "nproc" reports all the cores of the host machine,
    # not the 8 we reserved, so tell OpenMP how many threads to use.
    env = dict(os.environ, OMP_NUM_THREADS=str(CPU_CORES), OMP_PROC_BIND="true",
               # ncu cannot lock the GPU clock speed inside the container.
               NCU_ARGS="--clock-control none")

    # Nsight Compute is installed outside the normal PATH.
    ncu = subprocess.run("ls /opt/nvidia/nsight-compute/*/ncu | head -1", shell=True,
                         capture_output=True, text=True).stdout.strip()
    if ncu:
        env["PATH"] = os.path.dirname(ncu) + ":" + env["PATH"]

    def sh(command: str, must_work: bool = True) -> None:
        print(f"\n$ {command}", flush=True)
        result = subprocess.run(command, shell=True, cwd="/repo", env=env)
        if must_work and result.returncode != 0:
            raise RuntimeError(f"`{command}` failed")

    sh("nvidia-smi")
    sh("bash bench/run_benchmarks.sh")
    sh("bash bench/profile_ncu.sh", must_work=False)   # profiling may be blocked in a container
    sh("python3 plots/plot_results.py")

    # Send every file in results/ back.
    return {p.name: p.read_bytes() for p in pathlib.Path("/repo/results").iterdir() if p.is_file()}


@app.local_entrypoint()
def main():
    files = run.remote()
    (ROOT / "results").mkdir(exist_ok=True)
    for name, data in files.items():
        (ROOT / "results" / name).write_bytes(data)
        print("wrote results/" + name)
