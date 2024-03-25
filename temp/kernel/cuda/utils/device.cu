#include <cuda_runtime.h>
#include <stdio.h>
#include "device.cuh"



EXPORT void print_device() {
    cudaDeviceProp prop;
    int count;
    cudaDeviceSynchronize();
    cudaGetDeviceCount(&count);

    printf("cudaGetDeviceCount:%d",count);

    for (int i = 0; i < count; i++) {
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d:\n", i);
        printf("\tName: %s\n", prop.name);
        printf("\tTotal global memory: %lu bytes\n", prop.totalGlobalMem);
        printf("\tTotal shared memory per block: %lu bytes\n", prop.sharedMemPerBlock);
        printf("\tTotal constant memory: %lu bytes\n", prop.totalConstMem);
        printf("\tNumber of multiprocessors: %d\n", prop.multiProcessorCount);
        printf("\tMax threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("\tMax threads dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("\tMax grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("\tClock rate: %d kHz\n", prop.clockRate);
        printf("\tCompute capability: %d.%d\n", prop.major, prop.minor);
        printf("\tTexture alignment: %lu bytes\n", prop.textureAlignment);
        printf("\tDevice overlap: %s\n", (prop.deviceOverlap ? "Enabled" : "Disabled"));
        printf("\tConcurrent kernels: %s\n", (prop.concurrentKernels ? "Enabled" : "Disabled"));
        printf("\tECC enabled: %s\n", (prop.ECCEnabled ? "Yes" : "No"));
        printf("\tUnified addressing: %s\n", (prop.unifiedAddressing ? "Yes" : "No"));
        printf("\tMemory clock rate: %d kHz\n", prop.memoryClockRate);
        printf("\tMemory bus width: %d bits\n", prop.memoryBusWidth);
        printf("\tL2 cache size: %d bytes\n", prop.l2CacheSize);
        printf("\tMax resident threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("\tCompute mode: %s\n", (prop.computeMode == cudaComputeModeDefault ? "Default" :
                                                    (prop.computeMode == cudaComputeModeExclusive ? "Exclusive" :
                                                    (prop.computeMode == cudaComputeModeProhibited ? "Prohibited" :
                                                    (prop.computeMode == cudaComputeModeExclusiveProcess ? "Exclusive Process" : "Unknown")))));
        printf("\tPCI bus ID: %d\n", prop.pciBusID);
        printf("\tPCI device ID: %d\n", prop.pciDeviceID);
        printf("\tMemory clock rate: %d kHz\n", prop.memoryClockRate);
        printf("\tMemory bus width: %d bits\n", prop.memoryBusWidth);
        printf("\tMax threads per warp: %d\n", prop.warpSize);
        printf("\tMax registers per block: %d\n", prop.regsPerBlock);
        printf("\tMax shared memory per multiprocessor: %lu bytes\n", prop.sharedMemPerMultiprocessor);
        printf("\tMax threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("\tCompute capability: %d.%d\n", prop.major, prop.minor);
        printf("\tClock rate: %d kHz\n", prop.clockRate);
        printf("\tKernel execution timeout: %s\n", (prop.kernelExecTimeoutEnabled ? "Enabled" : "Disabled"));
    }
}