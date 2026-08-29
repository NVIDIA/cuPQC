NVCC=nvcc
NVCC_FLAGS=-dlto -arch=native -std=c++17 -O3

# cuPQC SDK paths - adjust these to match your installation
# Option 1: Set CUPQC_SDK_DIR environment variable
# Option 2: Modify CUPQC_SDK_DIR below to point to your SDK installation
# Common installation locations:
#   - /usr/local/cupqc-sdk (default, recommended)
#   - /opt/cupqc-sdk or any user-specified path
CUPQC_SDK_DIR ?= /usr/local/cupqc-sdk
CUPQC_INCLUDE_DIR ?= $(CUPQC_SDK_DIR)/include/cupqc
COMMONDX_INCLUDE_DIR ?= $(CUPQC_SDK_DIR)/include
CUPQC_LIB_DIR ?= $(CUPQC_SDK_DIR)/lib

# CUDA paths
CUDA_BIN_DIR=$(shell dirname `which $(NVCC)`)
CUDA_INCLUDE_DIR=$(CUDA_BIN_DIR)/../include

# Source and target
SRC = merkle_proof.cu
TARGET = merkle_proof

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -L$(CUPQC_LIB_DIR) -lcupqc-hash -o $@ $< -I$(COMMONDX_INCLUDE_DIR) -I$(CUPQC_INCLUDE_DIR)

.PHONY: all clean

all: $(TARGET)

clean:
	rm -f $(TARGET)

.DEFAULT_GOAL := all
