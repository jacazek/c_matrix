//
// Created by jacob on 9/23/24.
//
#include <stdio.h>
#include "gpu_matrix_multiply.h"
#include "cuda_runtime.h"

// CUDA kernel function for matrix multiplication
template<typename T>
__global__ void matrixMultiplyKernel(T *A, T *B, T *C, int l, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < l && col < n) {
        T result = 0;
        for (int i = 0; i < m; i++) {
            result += A[row * m + i] * B[i * n + col];
            // result = A[row * m + i];
        }
        C[row * n + col] = result;
    }
}

template<typename T>
__host__ void matrixMultiply(matrix_2d *A, matrix_2d *B, matrix_2d *C) {
    // number of columns in matrix A (row) and rows in matrix B
    int m = A->x_length;
    // number of rows (column) in matrix A
    int l = A->y_length;
    // number of columns in matrix B
    int n = B->x_length;

    int A_size = (A->x_length * A->y_length) * sizeof(T);
    printf("Data size: %i\n", A->data_size);
    int B_size = (B->x_length * B->y_length) * sizeof(T);
    int C_size = (C->x_length * C->y_length) * sizeof(T);

    T *d_A, *d_B, *d_C;

    cudaError_t result;
    result = cudaMalloc((void **) &d_A, A_size);
    if (result != cudaSuccess) {
        printf("CudaMalloc failed: %s\n", cudaGetErrorString(result));
    }
    result = cudaMalloc((void **) &d_B, B_size);
    if (result != cudaSuccess) {
        printf("CudaMalloc failed: %s\n", cudaGetErrorString(result));
    }
    result = cudaMalloc((void **) &d_C, C_size);
    if (result != cudaSuccess) {
        printf("CudaMalloc failed: %s\n", cudaGetErrorString(result));
    }

    result = cudaMemcpy(d_A, A->data, A_size, cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        printf("CudaMalloc failed: %s\n", cudaGetErrorString(result));
    }
    result = cudaMemcpy(d_B, B->data, B_size, cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        printf("CudaMalloc failed: %s\n", cudaGetErrorString(result));
    }

    // matrix multiply
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (l + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrixMultiplyKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, l, m, n);
    result = cudaPeekAtLastError();
    if (result != cudaSuccess) {
        printf("kernel run failed: %s\n", cudaGetErrorString(result));
    }

    cudaDeviceSynchronize();
    cudaMemcpy(C->data, d_C, C_size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

__host__ void matrix2D_gpu_matmul(matrix_2d *A, matrix_2d *B, matrix_2d *C) {
    if (A->precision == INT) {
        matrixMultiply<int>(A, B, C);
    } else if (A->precision == FLOAT) {
        matrixMultiply<float>(A, B, C);
    } else {
        matrixMultiply<double>(A, B, C);
    }
}
