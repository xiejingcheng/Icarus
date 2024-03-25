#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>

#include "cucumat_kernel.cuh"

#define cublas_check(call)                                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        cublasStatus_t err = call;                                                                                     \
        if (err != CUBLAS_STATUS_SUCCESS)                                                                              \
        {                                                                                                              \
            printf("cuBLAS Error: \n");                                                                                \
            printf("    File:       %s\n", __FILE__);                                                                  \
            printf("    Line:       %d\n", __LINE__);                                                                  \
            printf("    Error Code: %d\n", err);                                                                       \
                                                            \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0)

#define cuda_check(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t err = call;                                                                                        \
        if (err != cudaSuccess)                                                                                        \
        {                                                                                                              \
            printf("CUDA Error: \n");                                                                                  \
            printf("    File:       %s\n", __FILE__);                                                                  \
            printf("    Line:       %d\n", __LINE__);                                                                  \
            printf("    Error Code: %d\n", err);                                                                       \
            printf("    Error Text: %s\n", cudaGetErrorString(err));                                                   \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0)

typedef struct
{
    int m;
    int n;
    float *data;

} CMatrix;

cublasHandle_t handle;

struct ReciprocalFunctor
{
    __device__ float operator()(const float &x) const
    {
        return 1.0f / x;
    }
};

struct AbsFunctor
{
    __device__ float operator()(const float &x) const
    {
        return fabsf(x);
    }
};

struct LessThanFunctor
{
    __device__ float operator()(const float &x, const float &y) const
    {
        return x < y ? 1.0f : 0.0f;
    }
};

struct GreaterThanFunctor
{
    __device__ float operator()(const float &x, const float &y) const
    {
        return x > y ? 1.0f : 0.0f;
    }
};

struct EqualToFunctor
{
    __device__ float operator()(const float &x, const float &y) const
    {
        return x == y ? 1.0f : 0.0f;
    }
};

extern "C"
{

    bool cublas_create()
    {
        cublas_check(cublasCreate(&handle));
        return true;
    }

    bool cublas_destroy()
    {
        cublas_check(cublasDestroy(handle));
        return true;
    }

    bool build_matrix_empty(int m, int n, CMatrix *mat)
    {
        mat->m = m;
        mat->n = n;
        int size = m * n * sizeof(float);
        cuda_check(cudaMalloc(&mat->data, size));
        return true;
    }

    bool build_matrix_with_fill(int m, int n, CMatrix *mat, float val)
    {
        mat->m = m;
        mat->n = n;
        int size = m * n * sizeof(float);
        int N = m * n;
        cuda_check(cudaMalloc(&mat->data, size));
        thrust::device_ptr<float> dev_ptr(mat->data);
        thrust::fill(dev_ptr, dev_ptr + N, val);
        return true;
    }

    bool build_matrix_from_array(int m, int n, CMatrix *mat, float *arr)
    {
        mat->m = m;
        mat->n = n;
        int size = m * n * sizeof(float);
        cuda_check(cudaMalloc(&mat->data, size));
        cuda_check(cudaMemcpy(mat->data, arr, size, cudaMemcpyHostToDevice));
        return true;
    }

    bool to_host(CMatrix *mat, float *arr)
    {
        int size = mat->m * mat->n * sizeof(float);
        cuda_check(cudaMemcpy(arr, mat->data, size, cudaMemcpyDeviceToHost));
        return true;
    }

    bool free_device_memory(CMatrix *mat)
    {
        if (mat->data != nullptr)
        {
            cuda_check(cudaFree(mat->data));
        }
        return true;
    }

    bool assign(CMatrix *mat, float *arr, bool from_host)
    {
        int size = mat->m * mat->n * sizeof(float);
        cudaMemcpyKind kind = cudaMemcpyDeviceToDevice;
        if (from_host)
        {
            kind = cudaMemcpyHostToDevice;
        }
        cuda_check(cudaMemcpy(mat->data, arr, size, kind));
        return true;
    }

    bool fill(CMatrix *mat, float val)
    {
        int N = mat->m * mat->n;
        thrust::device_ptr<float> dev_ptr(mat->data);
        thrust::fill(dev_ptr, dev_ptr + N, val);
        return true;
    }

    bool copy(CMatrix *src, CMatrix *dst)
    {
        dst->m = src->m;
        dst->n = src->n;
        int size = src->m * src->n * sizeof(float);
        cuda_check(cudaMalloc(&dst->data, size));
        cuda_check(cudaMemcpy(dst->data, src->data, size, cudaMemcpyDeviceToDevice));
        return true;
    }

    bool matrix_abs(CMatrix *src, CMatrix *dst)
    {
        int N = src->m * src->n;
        thrust::device_ptr<float> first(src->data);
        thrust::device_ptr<float> last = first + N;
        thrust::device_ptr<float> result(dst->data);
        thrust::transform(first, last, result, AbsFunctor());
        return true;
    }

    bool matrix_negative(CMatrix *src, CMatrix *dst)
    {
        int N = src->m * src->n;
        thrust::device_ptr<float> first(src->data);
        thrust::device_ptr<float> last = first + N;
        thrust::device_ptr<float> result(dst->data);
        thrust::transform(first, last, result, thrust::negate<float>());
        return true;
    }

    bool matrix_reciprocal(CMatrix *src, CMatrix *dst)
    {
        int N = src->m * src->n;
        thrust::device_ptr<float> first(src->data);
        thrust::device_ptr<float> last = first + N;
        thrust::device_ptr<float> result(dst->data);
        thrust::transform(first, last, result, ReciprocalFunctor());
        return true;
    }

    bool matrix_add_matrix(CMatrix *src, CMatrix *dst, CMatrix *val)
    {
        int N = src->m * src->n;
        thrust::device_ptr<float> first1(src->data);
        thrust::device_ptr<float> last1 = first1 + N;
        thrust::device_ptr<float> first2(val->data);
        thrust::device_ptr<float> result(dst->data);
        thrust::transform(first1, last1, first2, result, thrust::plus<float>());
        return true;
    }

    bool matrix_add_scalar(CMatrix *src, CMatrix *dst, float val)
    {
        int N = src->m * src->n;
        thrust::device_ptr<float> first1(src->data);
        thrust::device_ptr<float> last1 = first1 + N;
        thrust::constant_iterator<float> first2(val);
        thrust::device_ptr<float> result(dst->data);
        thrust::transform(first1, last1, first2, result, thrust::plus<float>());
        return true;
    }

    bool matrix_sub_matrix(CMatrix *src, CMatrix *dst, CMatrix *val)
    {
        int N = src->m * src->n;
        thrust::device_ptr<float> first1(src->data);
        thrust::device_ptr<float> last1 = first1 + N;
        thrust::device_ptr<float> first2(val->data);
        thrust::device_ptr<float> result(dst->data);
        thrust::transform(first1, last1, first2, result, thrust::minus<float>());
        return true;
    }

    bool matrix_sub_scalar(CMatrix *src, CMatrix *dst, float val)
    {
        int N = src->m * src->n;
        thrust::device_ptr<float> first1(src->data);
        thrust::device_ptr<float> last1 = first1 + N;
        thrust::constant_iterator<float> first2(val);
        thrust::device_ptr<float> result(dst->data);
        thrust::transform(first1, last1, first2, result, thrust::minus<float>());
        return true;
    }

    bool matrix_mul_matrix(CMatrix *src, CMatrix *dst, CMatrix *val)
    {
        int N = src->m * src->n;
        thrust::device_ptr<float> first1(src->data);
        thrust::device_ptr<float> last1 = first1 + N;
        thrust::device_ptr<float> first2(val->data);
        thrust::device_ptr<float> result(dst->data);
        thrust::transform(first1, last1, first2, result, thrust::multiplies<float>());
        return true;
    }

    bool matrix_mul_scalar(CMatrix *src, CMatrix *dst, float val)
    {
        int N = src->m * src->n;
        thrust::device_ptr<float> first1(src->data);
        thrust::device_ptr<float> last1 = first1 + N;
        thrust::constant_iterator<float> first2(val);
        thrust::device_ptr<float> result(dst->data);
        thrust::transform(first1, last1, first2, result, thrust::multiplies<float>());
        return true;
    }

    bool matrix_div_matrix(CMatrix *src, CMatrix *dst, CMatrix *val)
    {
        int N = src->m * src->n;
        thrust::device_ptr<float> first1(src->data);
        thrust::device_ptr<float> last1 = first1 + N;
        thrust::device_ptr<float> first2(val->data);
        thrust::device_ptr<float> result(dst->data);
        thrust::transform(first1, last1, first2, result, thrust::divides<float>());
        return true;
    }

    bool matrix_div_scalar(CMatrix *src, CMatrix *dst, float val)
    {
        int N = src->m * src->n;
        thrust::device_ptr<float> first1(src->data);
        thrust::device_ptr<float> last1 = first1 + N;
        thrust::constant_iterator<float> first2(val);
        thrust::device_ptr<float> result(dst->data);
        thrust::transform(first1, last1, first2, result, thrust::divides<float>());
        return true;
    }

    bool view_cols(CMatrix *src, CMatrix *dst, int start, int end)
    {
        dst->m = src->m;
        dst->n = end - start;
        dst->data = src->data + start * src->m;
        return true;
    }

    bool get_rows(CMatrix *src, CMatrix *dst, int start, int end)
    {
        dst->m = end - start;
        dst->n = src->n;
        int size = dst->m * dst->n * sizeof(float);
        cuda_check(cudaMalloc(&dst->data, size));
        dim3 block_size(WARP_SIZE, WARP_SIZE);
        dim3 grid_size((int)ceil(1.0 * dst->m / WARP_SIZE), (int)ceil(1.0 * dst->n / WARP_SIZE));
        get_rows_kernel<<<grid_size, block_size>>>(src->m, src->n, src->data, start, end, dst->data);
        cuda_check(cudaGetLastError());
        return true;
    }

    bool set_rows(CMatrix *src, CMatrix *dst, int start, int end)
    {
        dim3 block_size(WARP_SIZE, WARP_SIZE);
        dim3 grid_size((int)ceil(1.0 * src->m / WARP_SIZE), (int)ceil(1.0 * src->n / WARP_SIZE));
        set_rows_kernel<<<grid_size, block_size>>>(dst->m, dst->n, dst->data, start, end, src->data);
        cuda_check(cudaGetLastError());
        return true;
    }

    bool transpose(CMatrix *src, CMatrix *dst)
    {
        dim3 block_size(WARP_SIZE, WARP_SIZE);
        dim3 grid_size((int)ceil(1.0 * src->m / WARP_SIZE), (int)ceil(1.0 * src->n / WARP_SIZE));
        transpose_kernel<<<grid_size, block_size>>>(src->m, src->n, src->data, dst->data);
        cuda_check(cudaGetLastError());
        return true;
    }

    bool axis_zero_sum(CMatrix *src, CMatrix *dst)
    {
        const int MAX_THREADS_AND_BLOCKS = 1024;
        int block_size = min(MAX_THREADS_AND_BLOCKS, src->n);
        int grid_size = min(MAX_THREADS_AND_BLOCKS, (int)ceil(1.0 * src->n / MAX_THREADS_AND_BLOCKS));
        axis_zero_sum_kernel<<<grid_size, block_size>>>(src->m, src->n, src->data, dst->data);
        return true;
    }

    bool all_sum(CMatrix *mat, float *result)
    {
        int N = mat->m * mat->n;
        thrust::device_ptr<float> first(mat->data);
        thrust::device_ptr<float> last = first + N;
        *result = thrust::reduce(first, last);
        return true;
    }

    bool less_than_matrix(CMatrix *src, CMatrix *dst, CMatrix *val)
    {
        int N = src->m * src->n;
        thrust::device_ptr<float> first1(src->data);
        thrust::device_ptr<float> last1 = first1 + N;
        thrust::device_ptr<float> first2(val->data);
        thrust::device_ptr<float> result(dst->data);
        thrust::transform(first1, last1, first2, result, LessThanFunctor());
        return true;
    }

    bool less_than_scalar(CMatrix *src, CMatrix *dst, float val)
    {
        int N = src->m * src->n;
        thrust::device_ptr<float> first1(src->data);
        thrust::device_ptr<float> last1 = first1 + N;
        thrust::constant_iterator<float> first2(val);
        thrust::device_ptr<float> result(dst->data);
        thrust::transform(first1, last1, first2, result, LessThanFunctor());
        return true;
    }

    bool greater_than_matrix(CMatrix *src, CMatrix *dst, CMatrix *val)
    {
        int N = src->m * src->n;
        thrust::device_ptr<float> first1(src->data);
        thrust::device_ptr<float> last1 = first1 + N;
        thrust::device_ptr<float> first2(val->data);
        thrust::device_ptr<float> result(dst->data);
        thrust::transform(first1, last1, first2, result, GreaterThanFunctor());
        return true;
    }

    bool greater_than_scalar(CMatrix *src, CMatrix *dst, float val)
    {
        int N = src->m * src->n;
        thrust::device_ptr<float> first1(src->data);
        thrust::device_ptr<float> last1 = first1 + N;
        thrust::constant_iterator<float> first2(val);
        thrust::device_ptr<float> result(dst->data);
        thrust::transform(first1, last1, first2, result, GreaterThanFunctor());
        return true;
    }

    bool equal_to_matrix(CMatrix *src, CMatrix *dst, CMatrix *val)
    {
        int N = src->m * src->n;
        thrust::device_ptr<float> first1(src->data);
        thrust::device_ptr<float> last1 = first1 + N;
        thrust::device_ptr<float> first2(val->data);
        thrust::device_ptr<float> result(dst->data);
        thrust::transform(first1, last1, first2, result, EqualToFunctor());
        return true;
    }

    bool equal_to_scalar(CMatrix *src, CMatrix *dst, float val)
    {
        int N = src->m * src->n;
        thrust::device_ptr<float> first1(src->data);
        thrust::device_ptr<float> last1 = first1 + N;
        thrust::constant_iterator<float> first2(val);
        thrust::device_ptr<float> result(dst->data);
        thrust::transform(first1, last1, first2, result, EqualToFunctor());
        return true;
    }

    bool axis_zero_min(CMatrix *src, CMatrix *dst)
    {
        const int MAX_THREADS_AND_BLOCKS = 1024;
        int block_size = min(MAX_THREADS_AND_BLOCKS, src->n);
        int grid_size = min(MAX_THREADS_AND_BLOCKS, (int)ceil(1.0 * src->n / MAX_THREADS_AND_BLOCKS));
        axis_zero_min_kernel<<<grid_size, block_size>>>(src->m, src->n, src->data, dst->data);
        return true;
    }

    bool all_min(CMatrix *mat, float *result)
    {
        int N = mat->m * mat->n;
        thrust::device_ptr<float> first(mat->data);
        thrust::device_ptr<float> last = first + N;
        thrust::device_ptr<float> tmp_result = thrust::min_element(first, last);
        *result = *tmp_result;
        return true;
    }

    bool axis_zero_max(CMatrix *src, CMatrix *dst)
    {
        const int MAX_THREADS_AND_BLOCKS = 1024;
        int block_size = min(MAX_THREADS_AND_BLOCKS, src->n);
        int grid_size = min(MAX_THREADS_AND_BLOCKS, (int)ceil(1.0 * src->n / MAX_THREADS_AND_BLOCKS));
        axis_zero_max_kernel<<<grid_size, block_size>>>(src->m, src->n, src->data, dst->data);
        return true;
    }

    bool all_max(CMatrix *mat, float *result)
    {
        int N = mat->m * mat->n;
        thrust::device_ptr<float> first(mat->data);
        thrust::device_ptr<float> last = first + N;
        thrust::device_ptr<float> tmp_result = thrust::max_element(first, last);
        *result = *tmp_result;
        return true;
    }

    bool dot(CMatrix *mat1, CMatrix *mat2, CMatrix *target_mat)
    {
        int m = mat1->m;
        int k = mat1->n;
        int n = mat2->n;

        float alpha = 1.0f;
        float beta = 1.0f;

        if (n == 1) // gemv if second matrix is a (column) vector
        {
            cublas_check(cublasSgemv(handle, CUBLAS_OP_N, m, k, &alpha, mat1->data, m, mat2->data, 1, &beta,
                                     target_mat->data, 1));
        }
        else if (m == 1) // gemv if first matrix is a (row) vector
        {
            cublas_check(cublasSgemv(handle, CUBLAS_OP_T, k, n, &alpha, mat2->data, k, mat1->data, 1, &beta,
                                     target_mat->data, 1));
        }
        else // gemm otherwise
        {
            cublas_check(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, mat1->data, m, mat2->data, k,
                                     &beta, target_mat->data, m));
        }
        return true;
    }
}
