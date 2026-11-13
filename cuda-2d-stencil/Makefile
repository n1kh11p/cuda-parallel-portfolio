# Portable build: override CUDA_HOME and CUDA_ARCH (e.g. 75, 80, 86) for your GPU.
CXXFLAGS ?= -std=c++11
CUDA ?= nvcc
CUDAFLAGS ?= -std=c++11 -O2 -Xcompiler "$(CXXFLAGS)"

CUDA_HOME ?= /usr/local/cuda
CUDA_ARCH ?= native

GENCODE := --generate-code arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH)
ifeq ($(CUDA_ARCH),native)
  GENCODE := -arch=native
endif

TARGET = 2d-stencil

all: $(TARGET)

$(TARGET): $(TARGET).cu
	$(CUDA) $(CUDAFLAGS) $(GENCODE) -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: all clean
