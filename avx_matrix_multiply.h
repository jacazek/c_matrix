//
// Created by jacob on 9/23/24.
//

#ifndef MATRIX_AVX_MATRIX_MULTIPLY_H
#define MATRIX_AVX_MATRIX_MULTIPLY_H
#include "matrix.h"
/**
 * Multiply a matrix by loading vectors of elements from row A and column B into AVX SIMD registers
 * Parallelizes multiplication of the elements of the vectors up to size 256.  So for doubles, you can
 * multiply 4 in parallel. For int 32 you can multiply 8 in parallel
 *
 * Requires your CPU to support AVX2 instructions
 * @param A Matrix A to multiply
 * @param B Matrix B to multiply
 * @param C Matrix in which result is accumulated
 */
void avx_matmul(matrix_2d *A, matrix_2d *B, matrix_2d *C);
#endif //MATRIX_AVX_MATRIX_MULTIPLY_H
