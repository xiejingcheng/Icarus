#include "gemm.cu"


int main() {
    // 设置矩阵维度
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;
    constexpr int BLOCK = 16;

    // 分配存储空间并初始化输入矩阵 A、B
    float *A_host = (float*)malloc(M * K * sizeof(float));
    float *B_host = (float*)malloc(K * N * sizeof(float));
    float *C_host = (float*)malloc(M * N * sizeof(float));

    // 初始化输入矩阵 A、B
    for (int i = 0; i < M * K; ++i) {
        A_host[i] = 1.0f;  // 可以根据需要自定义初始化值
    }
    for (int i = 0; i < K * N; ++i) {
        B_host[i] = 2.0f;  // 可以根据需要自定义初始化值
    }

    // 在 GPU 上分配内存并将输入数据传输到 GPU
    float *A_device, *B_device, *C_device;
    cudaMalloc((void**)&A_device, M * K * sizeof(float));
    cudaMalloc((void**)&B_device, K * N * sizeof(float));
    cudaMalloc((void**)&C_device, M * N * sizeof(float));

    cudaMemcpy(A_device, A_host, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B_host, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // 执行 GEMM 操作
    dim3 block(BLOCK, BLOCK);
    dim3 grid((M + BLOCK - 1) / BLOCK, (N + BLOCK - 1) / BLOCK);
    for(int i = 1;i<10;i++)
    sgemm<BLOCK><<<grid, block>>>(M, N, K, A_device, B_device, C_device);

    // 将结果从 GPU 拷贝回主机端
    cudaMemcpy(C_host, C_device, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 输出结果或进行其他操作

    // 释放内存
    free(A_host);
    free(B_host);
    free(C_host);
    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);

    return 0;
}
