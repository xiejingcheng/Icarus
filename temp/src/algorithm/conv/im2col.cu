//https://github.com/piojanu/CUDA-im2col-conv
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include<chrono>
#include <cublas_v2.h>

// Feature maps dimensionality descriptions and assumptions:
//             : Height          : Width           : Channels  : Number                    :
// INPUT   / A | H               | W               | C         | ------------------------- |  
// KERNELS / F | P = K           | Q = K           | R = C     | D = number of kernels = 1 |
// OUTPUT  / B | L = H * (K - 1) | M = W * (K - 1) | N = D = 1 | ------------------------- |
// [!] K must be odd number.
// [!] Data layout for INPUT/OUTPUT: C x H x W.
// [!] Data layout for KERNELS: D x R(=C) x P(=K) x Q(=K)

// Turn on/off debug mode
// #define DEBUG
// #define FUNCTEST
#define PERFTEST

#ifdef DEBUG
    #define LOG(...) printf(__VA_ARGS__); fflush(stdout);
#else
    #define LOG(...) ;
#endif

struct  
{
	__int64 tv_sec;
	__int64 tv_usec;
}timeval;


const unsigned int H = 256, W = 256, C = 80, K = 3; 

// HOST FUNCTION
// Takes matrix A [float *matA] and transforms it
// into column representation [float *matAc]
void im2colOnHost(float *matA, float *matAc, int radiusF, int countF, int L, int M, int K, int C)
{
    // For each spatial position in output...
    for (int m = 0; m < M; m++) {
        int w = m + radiusF;
        for (int l = 0; l < L; l++) {
            int h = l + radiusF;

            // Progress..
            LOG("\r[i] Calculation on CPU %3d%%...", ((m * L + l) * 100 / (M * L)));

            // For each kernel weight...
            for (int q = 0, oq = -1 * radiusF; oq <= radiusF; q++, oq++) {
                for (int p = 0, op = -1 * radiusF; op <= radiusF; p++, op++) {
                    for (int r = 0; r < C; r++) {
                        matAc[(r + C * (p + K * q)) + countF * (l + L * m)] = matA[r + C * ((h + op) + H * (w + oq))]; 
                        // LOG("matAc[%3d x %3d] <- matA[%3d x %3d x %3d]\n", (r + C * (p + K* q)), (l + L * m), (h + op), (w + oq), r);
                    }
                }
            }
        }
    }
    LOG("\n");
}
 
// DEVICE KERNEL
// Takes matrix A [float *matA] and transforms it
// into column representation [float *matAc] on GPU
__global__ 
void im2colOnDevice(unsigned int n, float *matAc, float *matA, int radiusF, int countF, int L, int M, int K, int C)
{
    // Using grid-stride loop if too big problem size.
    // https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < n; 
         idx += blockDim.x * gridDim.x) 
    {
        int m = (idx / C) / L;
        int l = (idx / C) % L;
        int r = idx % C;
        
        // For each spatial position in output...
        if (m < M) {
            int w = m + radiusF;
            if (l < L) {
                int h = l + radiusF;
                // For each kernel weight...
                for (int q = 0, oq = -1 * radiusF; oq <= radiusF; q++, oq++) {
                    for (int p = 0, op = -1 * radiusF; op <= radiusF; p++, op++) {
                        if (r < C) {
                            matAc[(r + C * (p + K * q)) + countF * (l + L * m)] = matA[r + C * ((h + op) + H * (w + oq))]; 
                        }
                    }
                }
            }
        }
    }
}
 
// DEVICE KERNEL
// Takes matrix A [float *matA] and transforms it
// into column representation [float *matAc] on GPU
__global__ 
void col2imOnDevice(unsigned int n, float *matA, float *matAc, int radiusF, int countF, int L, int M, int K, int C)
{
    // Using grid-stride loop if too big problem size.
    // https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < n; 
         idx += blockDim.x * gridDim.x) 
    {
        int m = (idx / C) / L;
        int l = (idx / C) % L;
        int r = idx % C;
    
        // For each spatial position in output...
        if (m < M) {
            int w = m + radiusF;
            if (l < L) {
                int h = l + radiusF;
                // For each kernel weight...
                for (int q = 0, oq = -1 * radiusF; oq <= radiusF; q++, oq++) {
                    for (int p = 0, op = -1 * radiusF; op <= radiusF; p++, op++) {
                        if (r < C) {
                            matA[r + C * ((h + op) + H * (w + oq))] = matAc[(r + C * (p + K * q)) + countF * (l + L * m)]; 
                        }
                    }
                }
            }
        }
    }
}

void program(unsigned int blockSize, unsigned int gridSize = 0)
{
    // CONSTS AND VARIABLES

    // Input/kernel/output counts and sizes
    const unsigned int countA = H*W*C;
    const size_t sizeA = countA*sizeof(float);
    LOG("[i] INPUT PARAMS: %u height, %u width, %u channels, %u elems, %u bytes\n", H, W, C, countA, sizeA);

    const unsigned int radiusF = (K - 1) / 2;
    const unsigned int countF = K*K*C;
    LOG("[i] FILTER PARAMS: %u radius, %u elems, %u bytes\n", radiusF, countF, countF*sizeof(float));
    
    const unsigned int L = H - (K - 1);
    const unsigned int M = W - (K - 1);
    LOG("[i] OUTPUT PARAMS: %u height, %u width, %u channels\n", L, M, 1);
    
    const unsigned int countLR = L * M;
    const unsigned int countAc = countF * countLR;
    const size_t sizeAc = countAc*sizeof(float);
    LOG("[i] INPUT IN COL PARAMS: %u elems, %u bytes\n", countAc, sizeAc);

    
    // PREPARE DATA

    // Generate input data
    float *matA = (float *)malloc(sizeA);
    for (int i = 0; i < countA; i++) {
        matA[i] = i;
    }
    LOG("  [!] FINISHED GENERATING INPUT\n");

#ifdef FUNCTEST
    // Calculate im2col result
    float *matAc = (float *)malloc(sizeAc);
    im2colOnHost(matA, matAc, radiusF, countF, L, M, K, C);
    LOG("  [!] FINISHED CALCULATING im2col RESULT ON CPU\n");
#endif


    // Alloc memory and copy data to device
    float *devA, *devAc, *retAc;
    
    cudaMalloc((void**)&devA, sizeA); 
    cudaMalloc((void**)&devAc, sizeAc); 
    retAc = (float *)malloc(sizeAc);

    cudaMemcpy(devA, matA, sizeA, cudaMemcpyHostToDevice); 

    // Compute default grid size if it wasn't passed
    const unsigned int KERNELS_NUM = L * M * C;
    if (gridSize == 0)
        gridSize = (KERNELS_NUM + blockSize - 1) / blockSize;
    
    // Run im2col computation on device and copy results
    im2colOnDevice<<<gridSize, blockSize>>>(KERNELS_NUM, devAc, devA, radiusF, countF, L, M, K, C);
    LOG("  [!] FINISHED CALCULATING im2col ON DEVICE\n");
    
    cudaMemcpy(retAc, devAc, sizeAc, cudaMemcpyDeviceToHost);

#ifdef FUNCTEST
    // Compare results
    int success = 1;
    for (int i = 0; i < countAc; i++) {
        if (retAc[i] != matAc[i]) {
            success = 0;
            printf("TEST FAILED: im2col device kernel...\n");
            break;
        }
    }

    if (success) {
        printf("TEST PASSED: im2col device kernel!\n");
    }
#endif

    // Allocate memory for return value
    float *retA;
    retA = (float *)malloc(sizeA);
    cudaMemset(devA, 0, sizeA); 
    
    // Run col2im computation on device and copy results
    col2imOnDevice<<<gridSize, blockSize>>>(KERNELS_NUM, devA, devAc, radiusF, countF, L, M, K, C);
    LOG("  [!] FINISHED CALCULATING col2im ON DEVICE\n");
    
    cudaMemcpy(retA, devA, sizeA, cudaMemcpyDeviceToHost);

#ifdef FUNCTEST
    // Compare results
    success = 1;
    for (int i = 0; i < countA; i++) {
        if (retA[i] != matA[i]) {
            success = 0;
            printf("TEST FAILED: col2im device kernel...\n");
            break;
        }
    }

    if (success) {
        printf("TEST PASSED: col2im device kernel!\n");
    }
#endif

    // CLEAN UP
    cudaFree(devA);
    cudaFree(devAc);
    
    free(matA);
#ifdef FUNCTEST
    free(matAc);
#endif
    free(retA);
    free(retAc);
}

int main()
{
    // Enforce default block and grid sizes
    unsigned int blockSize = 256;
    unsigned int gridSize = 0;

    // Calculate max needed kernels/threads number
    const unsigned int L = H - (K - 1);
    const unsigned int M = W - (K - 1);
    const unsigned int KERNELS_NUM = L * M * C;

    // Prepare variables for time measurement
    double elapsedTime, totalTime = 0;
    int totalRuns = 1;
    
    // First warm-up run
    LOG("--------- WARM-UP ---------\n");
    program(256);
    LOG("--------- WARM-UP ---------\n\n");

#ifdef PERFTEST
    // Average over 10 runs
    totalRuns = 10;
    
    // Open file for perf logs
    std::fstream fperflog("perflog.csv", std::ios::out);
    if (fperflog.good())
    {
        // Measure effect of different block sizes
        const unsigned int MAX_BLOCK_SIZE = 2048;
        for (blockSize = 1; blockSize <= MAX_BLOCK_SIZE; blockSize *= 2) {
            const unsigned int MAX_GRID_SIZE = (KERNELS_NUM + blockSize - 1) / blockSize;
            LOG("  [!] For %d blocks, max grid size is %d\n", blockSize, MAX_GRID_SIZE);
            for (gridSize = 1; gridSize <= 8192; gridSize *= 2) {
                if (gridSize <= MAX_GRID_SIZE) {
                    totalTime = 0;
                    for (int i = 0; i < totalRuns; i++)
#endif
                    {
                        // Start timer
                    
                        // WORK HARD!
                        program(blockSize, gridSize);
                    
                        // Stop timer

                    
                        // Compute the elapsed time in millisec

                    
                        totalTime += elapsedTime;
                    }
                    LOG("  [!] Whole program took %.3fms averaged over %d runs\n", totalTime / totalRuns, totalRuns);
#ifdef PERFTEST
                    fperflog << blockSize << "," << gridSize << "," << elapsedTime << std::endl;
                } else {
                    // Meaningless data, there is more grids ten data cat utilize 
                    fperflog << blockSize << "," << gridSize << "," << -1 << std::endl;
                }
            }
        }
        
        // Close file
        fperflog.close();
    }
#endif

    return EXIT_SUCCESS;
}