NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_70

all: main

main: main.cu neural_network.cu cuda_utils.cu data_loader.cu kernels.cu training.cu
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

clean:
	rm -f main
