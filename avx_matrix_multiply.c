//
// Created by jacob on 9/23/24.
//

#include "avx_matrix_multiply.h"
#include "immintrin.h"


void avx_double_matmul(matrix_2d *A, matrix_2d *B, matrix_2d *C, int m, int l, int n) {
    double *A_data = A->data;
    double *B_data = B->data;
    double *C_data = C->data;
    int blockSize = 4;
    int nonBlockElements = m % blockSize;

    // for row of matrix A
    for (int a_row = 0; a_row < l; a_row++) {
        // for column of matrix B
        for (int b_column = 0; b_column < n; b_column++) {
            // for each block in row/column of A/B
            // block here is of length 256
            // only works on CPUs with AVX2 instruction set available
            __m256d sum = _mm256_setzero_pd();
            for (int block = 0; block <= m - blockSize; block += blockSize) {
                // Multiply sub-blocks
                // for each element in matrix A block
                __m256d avx_a = _mm256_loadu_pd(&A_data[a_row * m + block]);
                __m256d avx_b = _mm256_loadu_pd(&B_data[b_column * m + block]);

                sum = _mm256_fmadd_pd(avx_a, avx_b, sum);
            }

            double remainder = 0.0;
            if (nonBlockElements) {
                // accumulate the remaining iteratively
                for (int element = m - nonBlockElements; element < m; element++) {
                    remainder += A_data[a_row * m + element] * B_data[b_column * m + element];
                }
            }

            double result[blockSize];
            _mm256_storeu_pd(result, sum);
            C_data[a_row * n + b_column] = result[0] + result[1] + result[2] + result[3] + remainder;
        }
    }
}

void avx_float_matmul(matrix_2d *A, matrix_2d *B, matrix_2d *C, int m, int l, int n) {
    float *A_data = A->data;
    float *B_data = B->data;
    float *C_data = C->data;
    int blockSize = 8;
    int nonBlockElements = m % blockSize;

    // for each row of matrix A
    for (int a_row = 0; a_row < l; a_row++) {
        // for each column of matrix B
        for (int b_column = 0; b_column < n; b_column++) {
            // for each block in row/column of A/B
            // block here is of length 256
            // only works on CPUs with AVX2 instruction set available
            __m256 sum = _mm256_setzero_ps();
            for (int block = 0; block <= m - blockSize; block += blockSize) {
                // Multiply sub-blocks
                // for each element in matrix A block
                __m256 avx_a = _mm256_loadu_ps(&A_data[a_row * m + block]);
                __m256 avx_b = _mm256_loadu_ps(&B_data[b_column * m + block]);

                sum = _mm256_fmadd_ps(avx_a, avx_b, sum);
            }

            float remainder = 0.0;
            if (nonBlockElements) {
                // accumulate the remaining iteratively
                for (int element = m - nonBlockElements; element < m; element++) {
                    remainder += A_data[a_row * m + element] * B_data[b_column * m + element];
                }
            }
            float result[blockSize];
            _mm256_storeu_ps(result, sum);
            C_data[a_row * n + b_column] = result[0] + result[1] + result[2] + result[3] + result[4] + result[5] +
                                           result[6] + result[7] + remainder;
        }
    }
}

void avx_int_matmul(matrix_2d *A, matrix_2d *B, matrix_2d *C, int m, int l, int n) {
    int *A_data = A->data;
    int *B_data = B->data;
    int *C_data = C->data;
    int blockSize = 8;
    int nonBlockElements = m % blockSize;

    // for each row of matrix A
    for (int a_row = 0; a_row < l; a_row++) {
        // for each column of matrix B
        for (int b_column = 0; b_column < n; b_column++) {
            // for each block in row/column of A/B
            // block here is of length 256
            // only works on CPUs with AVX2 instruction set available
            __m256i sum = _mm256_setzero_si256();
            for (int block = 0; block <= m - blockSize; block += blockSize) {
                // load blocksize sub-vectors of each matrix
                __m256i avx_a = _mm256_loadu_si256((__m256i *) &A_data[a_row * m + block]);
                __m256i avx_b = _mm256_loadu_si256((__m256i *) &B_data[b_column * m + block]);

                // multiply the two vectors of integers (elementwise)
                __m256i product = _mm256_mullo_epi32(avx_a, avx_b);

                // accumulate the product with the sum
                sum = _mm256_add_epi32(sum, product);
            }

            int remainder = 0;
            if (nonBlockElements) {
                // accumulate the remaining iteratively
                for (int element = m - nonBlockElements; element < m; element++) {
                    remainder += A_data[a_row * m + element] * B_data[b_column * m + element];
                }
            }
            // roll the results up to single element
            int result[blockSize];
            _mm256_storeu_si256((__m256i *) result, sum);
            C_data[a_row * n + b_column] = result[0] + result[1] + result[2] + result[3] + result[4] + result[5] +
                                           result[6] + result[7] + remainder;
        }
    }
}

void avx_matmul(matrix_2d *A, matrix_2d *B, matrix_2d *C) {
    check_matrix_compatibility(A, B, C);
    // clone and transpose the second matrix so that column values are sequential in memory
    // AVX instructions load sequential blocks of memory into registers. Since matrices
    // are by default row major (row values sequential), we transpose the second matrix as
    // matrix multiplication multiplies rows of A with columns of B.  Transposing makes the
    // matrix B column values sequential in memory and able to load into AVX registers.
    matrix_2d *B_copy = matrix2D_copy(B);
    matrix2D_transpose(B_copy);

    // number of columns in matrix A (row) and rows in matrix B
    int m = A->x_length;
    // number of rows (column) in matrix A
    int l = A->y_length;
    // number of columns in matrix B
    int n = B_copy->x_length;

    memset(C->data, 0, C->x_length * C->y_length * C->data_size);

    switch (A->precision) {
        case DOUBLE:
            avx_double_matmul(A, B_copy, C, m, l, n);
            break;
        case FLOAT:
            avx_float_matmul(A, B_copy, C, m, l, n);
            break;
        case INT:
            avx_int_matmul(A, B_copy, C, m, l, n);
            break;
    }
}
