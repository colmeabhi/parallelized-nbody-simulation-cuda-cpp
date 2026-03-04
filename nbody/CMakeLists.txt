cmake_minimum_required(VERSION 3.18)
project(galaxy_nbody LANGUAGES CXX CUDA)

# ── Language standards ────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD  14)
set(CMAKE_CUDA_STANDARD 14)

# ── CUDA compiler flags ───────────────────────────────────────────────────────
# -O3              : aggressive optimisation
# --use_fast_math  : faster but slightly less precise math (rsqrtf, sinf, …)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

# ── Executable ────────────────────────────────────────────────────────────────
add_executable(galaxy_nbody
    src/main.cu
    src/galaxy_simulation.cu
    src/nbody_kernel.cu
)

target_include_directories(galaxy_nbody PRIVATE src)

# ── GPU architecture targets ──────────────────────────────────────────────────
# Adjust this list to match your GPU (or set CUDA_ARCHITECTURES in the env).
# Common values:
#   60 = Pascal  (GTX 10xx)   75 = Turing  (RTX 20xx)
#   70 = Volta   (V100)       80 = Ampere  (A100, RTX 30xx)
#   86 = Ampere  (RTX 3080+)  89 = Ada     (RTX 40xx)
set_target_properties(galaxy_nbody PROPERTIES
    CUDA_ARCHITECTURES "60;70;75;80;86;89"
)
