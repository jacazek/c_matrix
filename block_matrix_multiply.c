//
// Created by jacob on 9/23/24.
//

#include "block_matrix_multiply.h"

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
                            double sum = C_data[y * n + x]; // Start with current value in C
                            for (int z = zz; z < zz + blockSize && z < m; z++) {
                                sum += A_data[y * m + z] * B_data[z * n + x];
                            }
                            C_data[y * n + x] = sum;
                        }
                    }
                }
            }
        }
    } else if (A->precision == FLOAT) {
        float *A_data = A->data;
        float *B_data = B->data;
        float *C_data = C->data;

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
                            float sum = C_data[y * n + x]; // Start with current value in C
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
                            int sum = C_data[y * n + x]; // Start with current value in C
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
