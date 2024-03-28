cmake_minimum_required(VERSION 3.9)
project(ica)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
find_package(CUDA REQUIRED)

#随着不同显卡类型更改
set(CUDA_NVCC_FLAGS -arch=sm_61 -O3)

set(SRC
    ${PROJECT_SOURCE_DIR}/cuda/ica.cu
    ${PROJECT_SOURCE_DIR}/cuda/utils/device.cu
    ${PROJECT_SOURCE_DIR}/cuda/operator/gemm.cu
    ${PROJECT_SOURCE_DIR}/cuda/oi/print.cu
    )
set(INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/cuda)

cuda_add_library(ica SHARED ${INCLUDE_DIRS} ${SRC})
