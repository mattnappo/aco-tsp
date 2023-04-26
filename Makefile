# the source files are in src/ directory
# the header files are in include/ directory
# the main program remains in the current directory

# CC   := g++
NVCC := nvcc

# the compiler flags
# CFLAGS     := -Wall -g -Iinclude -O3 # -fopenmp 
NVCC_FLAGS := -O3   -g -Iinclude -Xcompiler -fopenmp 
NVCC_LIBS := 

# cuda config
CUDA_ROOT_DIR=/usr/local/cuda-12.0
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
CUDA_LINK_LIBS= -lcudart

# source files in src/ directory
SRC := $(filter-out src/tests.cu src/cpu_main.cu src/gpu_main.cu, $(wildcard src/*.cu))
# CUDA_SRC := $(wildcard src/*.cu)

# object files in obj/ directory
OBJ      := $(patsubst src/%.cu,obj/%.o,$(SRC))
# CUDA_OBJ := $(patsubst src/%.cu,obj/%.o,$(CUDA_SRC))

# the executable file
TARGET := cpu
GPU_TARGET := gpu
OMP_TARGET := cpu_omp

# the default target
all: $(TARGET) $(GPU_TARGET) $(OMP_TARGET)

# the executable file depends on the object files
$(TARGET): $(OBJ) src/cpu_main.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# the executable file depends on the object files
$(OMP_TARGET): $(OBJ) src/cpu_main.cu
	$(NVCC) $(NVCC_FLAGS) -DUSE_OMP -o $@ $^

# the executable file depends on the object files
$(GPU_TARGET): $(OBJ) src/gpu_main.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

tests: $(OBJ) src/tests.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# the object files depend on the source files
obj/%.o: src/%.cu
	$(NVCC) $(NVCC_FLAGS) -dc -o $@ $<

# compile cuda objects
# obj/%.o: src/%.cu include/%.cuh
# 	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

# the clean target
clean:
	rm -f obj/* $(TARGET) $(GPU_TARGET) $(OMP_TARGET) tests sample_test.txt

# the run target
run: $(TARGET)
	./$(TARGET)

# the debug target
debug: $(TARGET)
	gdb $(TARGET)

# the valgrind target
valgrind: $(TARGET)
	valgrind --leak-check=full ./$(TARGET) data/ts11.tsp sols/ts11.sol

# the cppcheck target
cppcheck:
	cppcheck --enable=all --inconclusive --std=c++11 --suppress=missingIncludeSystem $(SRC)

