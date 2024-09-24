//
// Created by jacob on 9/23/24.
//

#ifndef MATRIX_BLOCK_MATRIX_MULTIPLY_H
#define MATRIX_BLOCK_MATRIX_MULTIPLY_H
#include "matrix.h"
/**
 * Multiply a matrix by loading blocks of data what will fit into fast CPU cache
 * @param A Matrix A to multiply
 * @param B Matrix B to multiply
 * @param C Matrix in which result is accumulated
 * @param blockSize The desired block size... should be adjusted to match YOUR CPU L1 cache size
 */
void block_matmul(matrix_2d *A, matrix_2d *B, matrix_2d *C, int blockSize);
#endif //MATRIX_BLOCK_MATRIX_MULTIPLY_H
