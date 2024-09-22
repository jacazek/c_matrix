//
// Created by jacob on 8/31/24.
//

#include "matrix.h"
#include <stdio.h>
#include "immintrin.h"
#define BLOCK_SIZE 64


void check_matrix_compatibility(const matrix_2d *A, const matrix_2d *B, const matrix_2d *C);

matrix_2d *matrix2D_new(MatrixPrecision precision, size_t x_length, size_t y_length) {
    matrix_2d *matrix = malloc(sizeof(matrix_2d));
    switch (precision) {
        case INT:
            matrix->data_size = sizeof(int);
            break;

        case DOUBLE:
            matrix->data_size = sizeof(double);
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
    }
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
    }

}

void naive_matmul(matrix_2d *A, matrix_2d *B, matrix_2d *C) {
    check_matrix_compatibility(A, B, C);

    int l = A->y_length;
    int m = A->x_length;
    int n = B->x_length;

    if (A->precision == DOUBLE) {
        double *A_data = A->data;
        double *B_data = B->data;
        double *C_data = C->data;

        for (int y = 0; y < l; y++) {
            for (int x = 0; x < n; x++) {
                int c_index = y * n + x;
                C_data[c_index] = 0.0;
                for (int z = 0; z < m; z++) {
                    C_data[c_index] += A_data[y * m + z] * B_data[z * n + x];
                }
            }
        }
    } else if (A->precision == INT) {
        int *A_data = A->data;
        int *B_data = B->data;
        int *C_data = C->data;

        for (int y = 0; y < l; y++) {
            for (int x = 0; x < n; x++) {
                int c_index = y * n + x;
                C_data[c_index] = 0;
                for (int z = 0; z < m; z++) {
                    C_data[c_index] += A_data[y * m + z] * B_data[z * n + x];
                }
            }
        }
    }
}

void check_matrix_compatibility(const matrix_2d *A, const matrix_2d *B,
                                const matrix_2d *C) {// TODO: check dimensions of A, B, and C are compatible
    if (A->x_length != B->y_length) { exit(1); }
    if (A->y_length != C->y_length) { exit(1); }
    if (B->x_length != C->x_length) { exit(1); }
}

void block_matmul(matrix_2d *A, matrix_2d *B, matrix_2d *C, int blockSize) {
    check_matrix_compatibility(A, B, C);


    // number of columns in matrix A (row) and rows in matrix B
    int m = A->x_length;
    // number of rows (column) in matrix A
    int l = A->y_length;
    // number of columns in matrix B
    int n = B->x_length;


    if (A->precision == DOUBLE) {
        double *A_data = A->data;
        double *B_data = B->data;
        double *C_data = C->data;

        // zero the output matrix
        for (int i = 0; i < l; i++) {
            for (int j = 0; j < n; j++) {
                C_data[i * n + j] = 0.0;
            }
        }

        // for each block of elements in row of matrix A
        for (int yy = 0; yy < l; yy += blockSize) {
            // for each block of elements in column of matrix B
            for (int xx = 0; xx < n; xx += blockSize) {
                // for each element
                for (int zz = 0; zz < m; zz += blockSize) {
                    // Multiply sub-blocks
                    // for each element in matrix A block
                    for (int y = yy; y < yy + blockSize && y < l; y++) {
                        for (int x = xx; x < xx + blockSize && x < n; x++) {
                            double sum = C_data[y * n + x];  // Start with current value in C
                            for (int z = zz; z < zz + blockSize && z < m; z++) {
                                sum += A_data[y * m + z] * B_data[z * n + x];
                            }
                            C_data[y * n + x] = sum;
                        }
                    }
                }
            }
        }
    } else if (A->precision == INT) {
        int *A_data = A->data;
        int *B_data = B->data;
        int *C_data = C->data;

        for (int i = 0; i < l; i++) {
            for (int j = 0; j < n; j++) {
                C_data[i * n + j] = 0;
            }
        }

        // for each block of elements in row of matrix A
        for (int yy = 0; yy < l; yy += blockSize) {
            // for each block of elements in column of matrix B
            for (int xx = 0; xx < n; xx += blockSize) {
                // for each element
                for (int zz = 0; zz < m; zz += blockSize) {
                    // Multiply sub-blocks
                    // for each element in matrix A block
                    for (int y = yy; y < yy + blockSize && y < l; y++) {
                        for (int x = xx; x < xx + blockSize && x < n; x++) {
                            int sum = C_data[y * n + x];  // Start with current value in C
                            for (int z = zz; z < zz + blockSize && z < m; z++) {
                                sum += A_data[y * m + z] * B_data[z * n + x];
                            }
                            C_data[y * n + x] = sum;
                        }
                    }
                }
            }
        }
    }
}

void avx_matmul(matrix_2d *A, matrix_2d *B, matrix_2d *C) {
    check_matrix_compatibility(A, B, C);
    matrix2D_transpose(B);

    // number of columns in matrix A (row) and rows in matrix B
    int m = A->x_length;
    // number of rows (column) in matrix A
    int l = A->y_length;
    // number of columns in matrix B
    int n = B->x_length;


    if (A->precision == DOUBLE) {
        double *A_data = A->data;
        double *B_data = B->data;
        double *C_data = C->data;

        // zero the output matrix
        for (int i = 0; i < l; i++) {
            for (int j = 0; j < n; j++) {
                C_data[i * n + j] = 0.0;
            }
        }
        int blockSize = 4;
        // for each block of elements in row of matrix A
        for (int a_row = 0; a_row < l; a_row++) {
            // for each block of elements in column of matrix B
            for (int b_column = 0; b_column < n; b_column++) {
                __m256d sum = _mm256_setzero_pd();

                // for each element
                for (int block = 0; block < m; block += blockSize) {
                    // Multiply sub-blocks
                    // for each element in matrix A block
                    __m256d avx_a = _mm256_loadu_pd(&A_data[a_row * m + block]);
                    __m256d avx_b = _mm256_loadu_pd(&B_data[block * n + b_column]);

                    sum = _mm256_fmadd_pd(avx_a, avx_b, sum);
                }
                double result[blockSize];
                _mm256_storeu_pd(result, sum);
                C_data[a_row * n + b_column] = result[0] + result[1] + result[2] + result[3];
            }
        }
    } else if (A->precision == INT) {
        int *A_data = A->data;
        int *B_data = B->data;
        int *C_data = C->data;

        for (int i = 0; i < l; i++) {
            for (int j = 0; j < n; j++) {
                C_data[i * n + j] = 0;
            }
        }

        int blockSize = 8;
        // for each block of elements in row of matrix A
        for (int a_row = 0; a_row < l; a_row++) {
            // for each block of elements in column of matrix B
            for (int b_column = 0; b_column < n; b_column++) {
                __m256i sum = _mm256_setzero_si256();

                // for each element
                for (int block = 0; block < m; block += blockSize) {
                    // load blocksize sub-vectors of each matrix
                    __m256i avx_a = _mm256_loadu_si256((__m256i*)&A_data[a_row * m + block]);
                    __m256i avx_b = _mm256_loadu_si256((__m256i*)&B_data[b_column * m + block]);

                    // multiply the two vectors of integers (elementwise)
                    __m256i product = _mm256_mullo_epi32(avx_a, avx_b);

                    // accumulate the product with the sum
                    sum = _mm256_add_epi32(sum, product);
                }
                // roll the results up to single element
                int result[blockSize];
                _mm256_storeu_si256((__m256i*)result, sum);
                C_data[a_row * n + b_column] = result[0] + result[1] + result[2] + result[3] + result[4] + result[5] + result[6] + result[7];
            }
        }
    }
}

void matrix_matmul(matrix_2d *A, matrix_2d *B, matrix_2d *C, MatmulStrategy strategy) {
    switch(strategy) {
        case NAIVE:
            naive_matmul(A, B, C);
            break;
        case BLOCK:
            block_matmul(A,B,C, 32);
            break;
        case AVX:
            avx_matmul(A,B,C);
            break;
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
        for (int i = 0; i< matrix_length; i++) {
            ((double*)matrix->data)[i] = ((double*)data)[i];
        }
    } else if (matrix->precision == INT) {
        for (int i = 0; i< matrix_length; i++) {
            ((int*)matrix->data)[i] = ((int*)data)[i];
        }
    }

}

void matrix2D_transpose(matrix_2d *matrix) {
    if (matrix->precision == DOUBLE) {
        double* temp = malloc(matrix->data_size * matrix->x_length * matrix->y_length);
        double* data = matrix->data;
        for (int i = 0; i < matrix->x_length; i++) {
            for (int j = 0; j < matrix->y_length; j++) {
                temp[j * matrix->x_length + i] = data[i * matrix->y_length + j];
            }
        }
        matrix->data = temp;
        free(data);
    }else if (matrix->precision == INT) {
        int* temp = malloc(matrix->data_size * matrix->x_length * matrix->y_length);
        int* data = matrix->data;
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
