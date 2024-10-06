//
// Created by jacob on 8/31/24.
//

#include "matrix.h"
#include <stdio.h>
#include "naive_memory_aligned_matrix_multiply.h"
#ifdef CUDA_SUPPORT
#include "gpu_matrix_multiply.h"
#endif
#ifdef AVX_SUPPORT
#include "avx_matrix_multiply.h"
#endif
#include "block_matrix_multiply.h"
#include "naive_matrix_multiply.h"
#define BLOCK_SIZE 64


matrix_2d *matrix2D_new(MatrixPrecision precision, size_t x_length, size_t y_length) {
    matrix_2d *matrix = malloc(sizeof(matrix_2d));
    switch (precision) {
        case INT:
            matrix->data_size = sizeof(int);
            break;
        case DOUBLE:
            matrix->data_size = sizeof(double);
            break;
        case FLOAT:
            matrix->data_size = sizeof(float);
            break;
        default:
            matrix->data_size = sizeof(int);
            break;
    }

    void *data = malloc(matrix->data_size * x_length * y_length);
    matrix->precision = precision;
    matrix->x_length = x_length;
    matrix->y_length = y_length;
    matrix->data = data;
    return matrix;
}

void matrix_zeros_naive(const matrix_2d *matrix) {
    memset(matrix->data, 0, matrix->data_size * matrix->x_length * matrix->y_length);
}

void matrix2D_get_element(matrix_2d *matrix, size_t x, size_t y, void *data) {
    int index = y * matrix->x_length + x;
    switch (matrix->precision) {
        case INT:
            *((int *) data) = ((int *) matrix->data)[index];
            break;
        case DOUBLE:
            *((double *) data) = ((double *) matrix->data)[index];
            break;
        case FLOAT:
            *((float *) data) = ((float *) matrix->data)[index];
    }
}

void matrix2D_set_element(matrix_2d *matrix, size_t x, size_t y, void *data) {
    int index = y * matrix->x_length + x;
    switch (matrix->precision) {
        case INT:
            ((int *) matrix->data)[index] = *(int *) data;
            break;
        case DOUBLE:
            ((double *) matrix->data)[index] = *((double *) data);
            break;
        case FLOAT:
            ((float *) matrix->data)[index] = *((float *) data);
    }
}

matrix_2d *matrix2D_copy(const matrix_2d *matrix) {
    void *copy_data = malloc(matrix->data_size * matrix->x_length * matrix->y_length);
    memcpy(copy_data, matrix->data, matrix->x_length * matrix->y_length * matrix->data_size);

    matrix_2d *copy_matrix = malloc(sizeof(matrix_2d));
    copy_matrix->precision = matrix->precision;
    copy_matrix->x_length = matrix->x_length;
    copy_matrix->y_length = matrix->y_length;
    copy_matrix->data_size = matrix->data_size;
    copy_matrix->data = copy_data;
    return copy_matrix;
}

void matrix_random(matrix_2d *matrix) {
    switch (matrix->precision) {
        case INT:
            for (int x = 0; x < matrix->x_length; x++) {
                for (int y = 0; y < matrix->y_length; y++) {
                    int value = (rand() % 2000 - 1000);
                    matrix2D_set_element(matrix, x, y, &value);
                }
            }
            break;
        case DOUBLE:
            for (int x = 0; x < matrix->x_length; x++) {
                for (int y = 0; y < matrix->y_length; y++) {
                    double value = (rand() % 2000 - 1000) / 1000.0;
                    matrix2D_set_element(matrix, x, y, &value);
                }
            }
            break;
        case FLOAT:
            for (int x = 0; x < matrix->x_length; x++) {
                for (int y = 0; y < matrix->y_length; y++) {
                    float value = (rand() % 2000 - 1000) / 1000.0;
                    matrix2D_set_element(matrix, x, y, &value);
                }
            }
            break;
    }
}


void check_matrix_compatibility(const matrix_2d *A, const matrix_2d *B,
                                const matrix_2d *C) {
    if (A->x_length != B->y_length) { exit(1); }
    if (A->y_length != C->y_length) { exit(1); }
    if (B->x_length != C->x_length) { exit(1); }
}

void matrix_matmul(matrix_2d *A, matrix_2d *B, matrix_2d *C, MatmulStrategy strategy) {
    switch (strategy) {
        case NAIVE:
            naive_matmul(A, B, C);
            break;
        case BLOCK:
            block_matmul(A, B, C, 64);
            break;
        case NAIVE_MEMORY_ALIGNED:
            naive_memory_aligned_matrix_multiply(A, B, C);
#ifdef AVX_SUPPORT
        case AVX:
            avx_matmul(A, B, C);
            break;
#endif
#ifdef CUDA_SUPPORT
        case GPU:
            matrix2D_gpu_matmul(A, B, C);
            break;
#endif
    }
}

void matrix2D_destroy(matrix_2d **matrix) {
    free((*matrix)->data);
    free(*matrix);
    matrix = NULL;
}

void matrix2D_fill(matrix_2d *matrix, void *data) {
    int matrix_length = matrix->x_length * matrix->y_length;
    if (matrix->precision == DOUBLE) {
        for (int i = 0; i < matrix_length; i++) {
            ((double *) matrix->data)[i] = ((double *) data)[i];
        }
    } else if (matrix->precision == INT) {
        for (int i = 0; i < matrix_length; i++) {
            ((int *) matrix->data)[i] = ((int *) data)[i];
        }
    } else if (matrix->precision == FLOAT) {
        for (int i = 0; i < matrix_length; i++) {
            ((float *) matrix->data)[i] = ((float *) data)[i];
        }
    }
}

void matrix2D_transpose(matrix_2d *matrix) {
    if (matrix->precision == DOUBLE) {
        double *temp = malloc(matrix->data_size * matrix->x_length * matrix->y_length);
        double *data = matrix->data;
        for (int i = 0; i < matrix->y_length; i++) {
            for (int j = 0; j < matrix->x_length; j++) {
                temp[j * matrix->y_length + i] = data[i * matrix->x_length + j];
            }
        }
        matrix->data = temp;
        free(data);
    } else if (matrix->precision == INT) {
        int *temp = malloc(matrix->data_size * matrix->x_length * matrix->y_length);
        int *data = matrix->data;
        for (int i = 0; i < matrix->y_length; i++) {
            for (int j = 0; j < matrix->x_length; j++) {
                temp[j * matrix->y_length + i] = data[i * matrix->x_length + j];
            }
        }
        matrix->data = temp;
        free(data);
    } else if (matrix->precision == FLOAT) {
        float *temp = malloc(matrix->data_size * matrix->x_length * matrix->y_length);
        float *data = matrix->data;
        for (int i = 0; i < matrix->y_length; i++) {
            for (int j = 0; j < matrix->x_length; j++) {
                temp[j * matrix->y_length + i] = data[i * matrix->x_length + j];
            }
        }
        matrix->data = temp;
        free(data);
    }
    size_t temp = matrix->x_length;
    matrix->x_length = matrix->y_length;
    matrix->y_length = temp;
}
