//
// Created by jacob on 9/23/24.
//
#include <stdio.h>
#include "gpu_matrix_multiply.h"
#include "cuda_runtime.h"

#define N 3  // Size of the matrices

// CUDA kernel function for matrix multiplication
template <typename T>
__global__ void matrixMultiply(T *A, T *B, T *C, int l, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < l && col < n) {
        int result = 0;
        for (int i = 0; i < m; i++) {
            result += A[row * m + i] * B[i * n + col];
        }
        C[row * n + col] = result;
    }
}

__host__ void matrix2D_gpu_matmul(matrix_2d *A, matrix_2d *B, matrix_2d *C) {
    // number of columns in matrix A (row) and rows in matrix B
    int m = A->x_length;
    // number of rows (column) in matrix A
    int l = A->y_length;
    // number of columns in matrix B
    int n = B->x_length;


    if (A->precision == INT) {
        int A_size = A->x_length * A->y_length *  A->data_size;
        int B_size = B->x_length * B->y_length * B->data_size;
        int C_size = C->x_length * C->y_length * C->data_size;

        int *d_A, *d_B, *d_C;

        cudaMalloc((void **) &d_A, A_size);
        cudaMalloc((void **) &d_B, B_size);
        cudaMalloc((void**) &d_C, C_size);

        cudaMemcpy(d_A, A->data, A_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B->data, B_size, cudaMemcpyHostToDevice);

        // matrix multiply
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (l + threadsPerBlock.y - 1) / threadsPerBlock.y);
        matrixMultiply<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, l, m, n);

        cudaDeviceSynchronize();

        cudaMemcpy(C->data, d_C, C_size, cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    } else {
        int A_size = A->x_length * A->y_length *  A->data_size;
        int B_size = B->x_length * B->y_length * B->data_size;
        int C_size = C->x_length * C->y_length * C->data_size;

        double *d_A, *d_B, *d_C;

        cudaMalloc((void **) &d_A, A_size);
        cudaMalloc((void **) &d_B, B_size);
        cudaMalloc((void**) &d_C, C_size);

        cudaMemcpy(d_A, A->data, A_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B->data, B_size, cudaMemcpyHostToDevice);

        // matrix multiply
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (l + threadsPerBlock.y - 1) / threadsPerBlock.y);
        matrixMultiply<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, l, m, n);

        cudaDeviceSynchronize();

        cudaMemcpy(C->data, d_C, C_size, cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
}