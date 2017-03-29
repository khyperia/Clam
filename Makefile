NVCC_ARGS=--use_fast_math --generate-line-info --gpu-architecture=sm_30

kernels: mandelbox.ptx mandelbrot.ptx

%.ptx: %.cu Makefile
	nvcc $(NVCC_ARGS) --ptx -o $@ $<
