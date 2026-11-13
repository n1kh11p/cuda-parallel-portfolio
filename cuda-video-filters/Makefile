# Portable build: override CUDA_HOME, OPENCV_CFLAGS, OPENCV_LIBS, or CUDA_ARCH as needed.
CXX ?= g++
CXXFLAGS ?= -std=c++11 -O2
CUDA ?= nvcc
CUDAFLAGS ?= -std=c++11 -O2 -Xcompiler "$(CXXFLAGS)"

CUDA_HOME ?= /usr/local/cuda
CUDA_ARCH ?= native

GENCODE := --generate-code arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH)
ifeq ($(CUDA_ARCH),native)
  GENCODE := -arch=native
endif

OPENCV_CFLAGS ?= $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv 2>/dev/null)
OPENCV_LIBS ?= $(shell pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv 2>/dev/null)

ifneq ($(OPENCV_CFLAGS),)
  CXX_INCLUDES := $(OPENCV_CFLAGS)
  CXX_LIBS := $(OPENCV_LIBS) -L$(CUDA_HOME)/lib64 -lcudart
else
  # Fallback: set OPENCV_PREFIX to your OpenCV install (include/opencv4 under PREFIX).
  OPENCV_PREFIX ?=
  CXX_INCLUDES := -I$(CUDA_HOME)/include -I$(OPENCV_PREFIX)/include/opencv4
  CXX_LIBS := -L$(OPENCV_PREFIX)/lib -L$(OPENCV_PREFIX)/lib64 -L$(CUDA_HOME)/lib64 \
              -lopencv_core -lopencv_videoio -lopencv_imgproc -lcudart
endif

TARGET = video-effect

all: $(TARGET)

video-effect.o: video-effect.cu
	$(CUDA) $(CUDAFLAGS) $(GENCODE) -o $@ -c $<

$(TARGET): driver.cpp video-effect.o
	$(CXX) $(CXXFLAGS) $(CXX_INCLUDES) -o $@ $^ $(CXX_LIBS)

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean
