//
// Created by jacob on 9/23/24.
//

#ifndef MATRIX_GPU_MATRIX_MULTIPLY_H
#define MATRIX_GPU_MATRIX_MULTIPLY_H
#include "matrix.h"

// cuda compiles a C++ library.  we need to explicitly make the library available for C
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Multiply a matrix by sending the matrix to the GPU and parallelizing multiplication of every element in the matrix
 * This code does not take into account whether the matrix fits in GPU memory.
 * @param A Matrix A to multiply
 * @param B Matrix B to multiply
 * @param C Matrix in which result is accumulated
 */
void matrix2D_gpu_matmul(matrix_2d *A, matrix_2d *B, matrix_2d *C);
#ifdef __cplusplus
}
#endif

#endif //MATRIX_GPU_MATRIX_MULTIPLY_H
