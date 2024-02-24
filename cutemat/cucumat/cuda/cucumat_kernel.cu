#include "cucumat_kernel.cuh"

#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>

__global__ void get_rows_kernel(int m, int n, float *src, int start, int end, float *dst)
{
    int src_row = start + blockIdx.x * blockDim.x + threadIdx.x;
    int src_col = blockIdx.y * blockDim.y + threadIdx.y;
    int dst_row = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_col = blockIdx.y * blockDim.y + threadIdx.y;

    if (src_row < end && src_col < n)
    {
        dst[dst_col * (end - start) + dst_row] = src[src_col * m + src_row];
    }
}

__global__ void set_rows_kernel(int m, int n, float *dst, int start, int end, float *src)
{
    int src_row = blockIdx.x * blockDim.x + threadIdx.x;
    int src_col = blockIdx.y * blockDim.y + threadIdx.y;
    int dst_row = start + blockIdx.x * blockDim.x + threadIdx.x;
    int dst_col = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_row < end && dst_col < n)
    {
        dst[dst_col * m + dst_row] = src[src_col * (end - start) + src_row];
    }
}

__global__ void transpose_kernel(int m, int n, float *src, float *dst)
{
    __shared__ float s_mem[WARP_SIZE][WARP_SIZE + 1];
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < n)
    {
        s_mem[threadIdx.x][threadIdx.y] = src[col * m + row];
    }
    __syncthreads();
    row = blockIdx.y * blockDim.x + threadIdx.x;
    col = blockIdx.x * blockDim.y + threadIdx.y;
    if (row < n && col < m)
    {
        dst[col * n + row] = s_mem[threadIdx.y][threadIdx.x];
    }
}

__global__ void axis_zero_sum_kernel(int m, int n, float *src, float *dst)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int skip = gridDim.x + blockDim.x;
    for (int col = tid; col < n; col += skip)
    {
        float *first = src + col * m;
        float *last = first + m;
        dst[col] = thrust::reduce(thrust::device, first, last);
    }
}

__global__ void axis_zero_min_kernel(int m, int n, float *src, float *dst)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int skip = gridDim.x + blockDim.x;
    for (int col = tid; col < n; col += skip)
    {
        float *first = src + col * m;
        float *last = first + m;
        float *tmp_result = thrust::min_element(thrust::device, first, last);
        dst[col] = *tmp_result;
    }
}

__global__ void axis_zero_max_kernel(int m, int n, float *src, float *dst)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int skip = gridDim.x + blockDim.x;
    for (int col = tid; col < n; col += skip)
    {
        float *first = src + col * m;
        float *last = first + m;
        float *tmp_result = thrust::max_element(thrust::device, first, last);
        dst[col] = *tmp_result;
    }
}