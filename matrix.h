//
// Created by jacob on 8/31/24.
//

#ifndef MATRIX_MATRIX_H
#define MATRIX_MATRIX_H
#include <string.h>

#include "stdlib.h"

typedef enum MATRIX_PRECISION {
    INT,
    DOUBLE,
    FLOAT
} MatrixPrecision;

typedef enum MATMUL_STRATEGY {
    NAIVE,
    BLOCK,
    AVX,
    GPU
} MatmulStrategy;

typedef struct matrix_2d {
    MatrixPrecision precision;
    size_t data_size;
    size_t x_length;
    size_t y_length;
    void *data;
} matrix_2d;


matrix_2d *matrix2D_new(MatrixPrecision precision, size_t x_length, size_t y_length);


void matrix_zeros_naive(const matrix_2d *matrix);

void matrix2D_get_element(matrix_2d *matrix, size_t x, size_t y, void *data);

void matrix2D_set_element(matrix_2d *matrix, size_t x, size_t y, void *data);

matrix_2d *matrix2D_copy(const matrix_2d *matrix);

void matrix2D_fill(matrix_2d *matrix, void *data_array);

void matrix_random(matrix_2d *matrix);

void matrix2D_transpose(matrix_2d *matrix);

void matrix_matmul(matrix_2d *A, matrix_2d *B, matrix_2d *C, MatmulStrategy strategy);

void check_matrix_compatibility(const matrix_2d *A, const matrix_2d *B, const matrix_2d *C);

void matrix2D_destroy(matrix_2d **matrix);
#endif //MATRIX_MATRIX_H
