# Standalone build of the Alpamayo preprocessing kernels: CUDA toolkit only.
#
#   cmake -S kernels/standalone -B runs/scratch/standalone_build -DCMAKE_CUDA_ARCHITECTURES=120 \
#         -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc                                        # 5090
#   cmake -S kernels/standalone -B runs/scratch/standalone_build -DCMAKE_CUDA_ARCHITECTURES=110 \
#         -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc                                        # Thor
#   cmake --build runs/scratch/standalone_build
#
# The architecture must match the board (12.0 for the RTX 5090, 11.0 for the
# Jetson AGX Thor); there is no fat binary on purpose, so the number is stated.
# The compiler is named explicitly because a distribution nvcc on PATH (an
# older toolkit) is not the one that built the PyTorch extension.
cmake_minimum_required(VERSION 3.24)
project(alpamayo_preprocess_standalone LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(preprocess_standalone
  main.cpp
  tables.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../preprocess.cu)
target_include_directories(preprocess_standalone PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_options(preprocess_standalone PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_link_libraries(preprocess_standalone PRIVATE CUDA::cudart)
