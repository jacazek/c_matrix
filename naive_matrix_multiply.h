//
// Created by jacob on 9/23/24.
//

#ifndef MATRIX_NAIVE_MATRIX_MULTIPLY_H
#define MATRIX_NAIVE_MATRIX_MULTIPLY_H
#include "matrix.h"
/**
 * Multiply a matrix element by element in a naive manner.. slow as dirt but most straightforward
 * @param A Matrix A to multiply
 * @param B Matrix B to multiply
 * @param C Matrix in which result is accumulated
 */
void naive_matmul(matrix_2d *A, matrix_2d *B, matrix_2d *C);

#endif //MATRIX_NAIVE_MATRIX_MULTIPLY_H
