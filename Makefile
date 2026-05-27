# Build everything into build/.
#
#   make        - CPU and GPU programs
#   make cpu    - only the CPU programs (no CUDA needed)
#   make gpu    - only the GPU programs (needs nvcc)
#   make clean

CXX      = g++
CXXFLAGS = -O3 -march=native -ffast-math -fopenmp -Isrc
NVCC     = nvcc
# -Xcompiler -fopenmp : total_energy() in cpu_nbody.cpp uses OpenMP
# -lineinfo           : lets Nsight Compute show source lines
NVFLAGS  = -O3 --use_fast_math -lineinfo -Isrc -Xcompiler -fopenmp

CPU_CODE = src/galaxy.cpp src/cpu_nbody.cpp
GPU_CODE = src/gpu_nbody.cu

all: cpu gpu
cpu: build/bench_cpu build/energy_drift
gpu: build/galaxy_nbody build/bench_gpu build/check_gpu

build/bench_cpu: bench/bench_cpu.cpp $(CPU_CODE) src/*.h
	@mkdir -p build
	$(CXX) $(CXXFLAGS) bench/bench_cpu.cpp $(CPU_CODE) -o $@

build/energy_drift: bench/energy_drift.cpp $(CPU_CODE) src/*.h
	@mkdir -p build
	$(CXX) $(CXXFLAGS) bench/energy_drift.cpp $(CPU_CODE) -o $@

build/galaxy_nbody: src/main.cu $(GPU_CODE) $(CPU_CODE) src/*.h
	@mkdir -p build
	$(NVCC) $(NVFLAGS) src/main.cu $(GPU_CODE) $(CPU_CODE) -o $@

build/bench_gpu: bench/bench_gpu.cu $(GPU_CODE) $(CPU_CODE) src/*.h
	@mkdir -p build
	$(NVCC) $(NVFLAGS) bench/bench_gpu.cu $(GPU_CODE) $(CPU_CODE) -o $@

build/check_gpu: bench/check_gpu.cu $(GPU_CODE) $(CPU_CODE) src/*.h
	@mkdir -p build
	$(NVCC) $(NVFLAGS) bench/check_gpu.cu $(GPU_CODE) $(CPU_CODE) -o $@

clean:
	rm -rf build

.PHONY: all cpu gpu clean
