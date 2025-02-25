//
// Created by jacob on 2/24/25.
//
#include <hip/hip_runtime.h>
#include <stdio.h>
#include "gpu_matrix_multiply.h"

// HIP kernel function for matrix multiplication
template<typename T>
__global__ void matrixMultiplyKernel(T *_A, T *_B, T *_C, int l, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < l && col < n) {
        T result = 0;
        for (int i = 0; i < m; i++) {
            result += _A[row * m + i] * _B[i * n + col];
        }
        _C[row * n + col] = result;
    }
}

template<typename T>
__host__ void matrixMultiply(matrix_2d _A, matrix_2d _B, matrix_2d *C) {
    int m = _A->x_length;
    int l = _A->y_length;
    int n = _B->x_length;
    int A_size = _A->x_length * _A->y_length * sizeof(T);
    int B_size = _B->x_length * _B->y_length * sizeof(T);
    int C_size = C->x_length * C->y_length * sizeof(T);
    T *d_A, *d_B, *d_C;
    hipError_t result;

    result = hipMalloc((void**) &d_A, A_size);
    if (result != hipSuccess) {
        printf("hipMalloc failed\n");
    }

    result = hipMalloc((void**) &d_B, B_size);
    if (result != hipSuccess) {
        printf("hipMalloc failed\n");
    }

    result = hipMalloc((void**) &d_C, C_size);
    if (result != hipSuccess) {
        printf("hipMalloc failed\n");
    }

    result = hipMemcpy(d_A, _A->data, A_size, hipMemcpyHostToDevice);
    if (result != hipSuccess) {
        printf("hipMemcpy failed\n");
    }

    result = hipMemcpy(d_B, _B->data, B_size, hipMemcpyHostToDevice);
    if (result != hipSuccess) {
        printf("hipMemcpy failed\n");
    }

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (l + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMultiplyKernel<<<numBlocks, threadsPerBlock>>>(_A->data, _B->data, d_C, l, m, n);

    result = hipPeekAtLastError();
    if (result != hipSuccess) {
        printf("Kernel execution failed\n");
    }

    hipDeviceSynchronize();
    result = hipMemcpy(C->data, d_C, C_size, hipMemcpyDeviceToHost);
    if (result != hipSuccess) {
        printf("hipMemcpy failed\n");
    }

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
}

__host__ void matrix2D_gpu_matmul(matrix_2d _A, matrix_2d _B, matrix_2d *C) {
    if (_A->precision == INT) {
        matrixMultiply<int>(_A, _B, C);
    } else if (_A->precision == FLOAT) {
        matrixMultiply<float>(_A, _B, C);
    } else if (_A->precision == DOUBLE) {
        matrixMultiply<double>(_A, _B, C);
    } else {
        printf("Unsupported precision\n");
    }
}