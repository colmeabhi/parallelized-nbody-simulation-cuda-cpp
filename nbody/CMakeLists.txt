cmake_minimum_required(VERSION 3.18)
project(galaxy_nbody LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)

# ── CPU code (always built; no CUDA needed) ──────────────────────────────────
# -march=native lets the serial/OpenMP baselines use AVX2/AVX-512, which is
# the fair comparison against the GPU.  Disable with -DNBODY_NATIVE=OFF.
option(NBODY_NATIVE "Compile CPU code with -march=native" ON)

add_library(nbody_cpu STATIC src/galaxy_ics.cpp src/cpu_nbody.cpp)
target_include_directories(nbody_cpu PUBLIC src)
target_link_libraries(nbody_cpu PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(nbody_cpu PRIVATE -O3 -ffast-math)
if(NBODY_NATIVE)
    target_compile_options(nbody_cpu PRIVATE -march=native)
endif()

add_executable(nbody_bench_cpu src/bench_cpu.cpp)
target_link_libraries(nbody_bench_cpu PRIVATE nbody_cpu)
target_compile_options(nbody_bench_cpu PRIVATE -O3)

# ── CUDA code (built only if a CUDA compiler is found) ───────────────────────
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    # Build for the GPU in this machine unless the user chose architectures
    # (e.g. -DCMAKE_CUDA_ARCHITECTURES="75;86").  "native" needs CMake ≥ 3.24.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
            set(CMAKE_CUDA_ARCHITECTURES native)
        else()
            set(CMAKE_CUDA_ARCHITECTURES 75)
        endif()
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # -O3 / --use_fast_math: rsqrtf etc. map to fast hardware approximations.
    # -lineinfo: source-line attribution in Nsight Compute, no runtime cost.
    set(NBODY_CUDA_FLAGS -O3 --use_fast_math -lineinfo)

    add_library(nbody_gpu STATIC src/nbody_kernel.cu)
    target_link_libraries(nbody_gpu PUBLIC nbody_cpu)
    target_compile_options(nbody_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${NBODY_CUDA_FLAGS}>)

    add_executable(galaxy_nbody src/main.cu src/galaxy_simulation.cu)
    target_link_libraries(galaxy_nbody PRIVATE nbody_gpu)
    target_compile_options(galaxy_nbody PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${NBODY_CUDA_FLAGS}>)

    add_executable(nbody_bench_gpu src/bench_gpu.cu)
    target_link_libraries(nbody_bench_gpu PRIVATE nbody_gpu)
    target_compile_options(nbody_bench_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${NBODY_CUDA_FLAGS}>)
else()
    message(WARNING "No CUDA compiler found: building only the CPU targets "
                    "(nbody_bench_cpu).")
endif()
