#pragma once

#define WARP_SIZE 32

__global__ void get_rows_kernel(int m, int n, float *src, int start, int end, float *dst);
__global__ void set_rows_kernel(int m, int n, float *dst, int start, int end, float *src);
__global__ void transpose_kernel(int m, int n, float *src, float *dst);
__global__ void axis_zero_sum_kernel(int m, int n, float *src, float *dst);
__global__ void axis_zero_min_kernel(int m, int n, float *src, float *dst);
__global__ void axis_zero_max_kernel(int m, int n, float *src, float *dst);