//
// Created by jacob on 10/6/24.
//

#ifndef TRANSPOSE_MATRIX_MULTIPLY_H
#define TRANSPOSE_MATRIX_MULTIPLY_H
#include "matrix.h"
/**
 * Multiply a matrix element by element in a naive manner.. slow as dirt but most straightforward
 * @param A Matrix A to multiply
 * @param B Matrix B to multiply
 * @param C Matrix in which result is accumulated
 */
void naive_memory_aligned_matrix_multiply(matrix_2d *A, matrix_2d *B, matrix_2d *C);
#endif //TRANSPOSE_MATRIX_MULTIPLY_H
