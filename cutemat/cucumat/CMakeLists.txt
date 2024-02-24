cmake_minimum_required(VERSION 3.18)
project(cucumat)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
# set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin/nvcc")
# set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4")
find_package(CUDA REQUIRED)
# set(CUDA_NVCC_FLAGS -arch=sm_75 -O3)

# set(CMAKE_CUDA_HOST_COMPILER "D:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Professional\\VC\\Tools\\MSVC\\14.16.27023\\bin\\Hostx86\\x64\\cl.exe")
set(CMAKE_CUDA_HOST_COMPILER "D:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Professional\\VC\\Tools\\MSVC\\14.16.27023\\bin\\Hostx86\\x64\\cl.exe")
#set(CUDA_HOST_COMPILER "D:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Professional\\VC\\Tools\\MSVC\\14.16.27023\\bin\\Hostx86\\x64\\cl.exe")
set(SRC
    ${PROJECT_SOURCE_DIR}/cuda/cucumat.cu
    ${PROJECT_SOURCE_DIR}/cuda/cucumat_kernel.cu)
set(INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/cuda)

cuda_add_library(cucumat SHARED ${INCLUDE_DIRS} ${SRC})
set_property(TARGET cucumat PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_link_libraries(cucumat cublas cudnn)
