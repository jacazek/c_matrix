//
// Created by jacob on 10/6/24.
//

#include "naive_memory_aligned_matrix_multiply.h"

void naive_memory_aligned_matrix_multiply(matrix_2d *A, matrix_2d *B, matrix_2d *C) {
    check_matrix_compatibility(A, B, C);
    matrix_2d *B_copy = matrix2D_copy(B);
    matrix2D_transpose(B_copy);

    int l = A->y_length;
    int m = A->x_length;
    int n = B_copy->y_length;

    if (A->precision == DOUBLE) {
        double *A_data = A->data;
        double *B_data = B_copy->data;
        double *C_data = C->data;

        for (int y = 0; y < l; y++) {
            for (int x = 0; x < n; x++) {
                int c_index = y * n + x;
                C_data[c_index] = 0.0;
                for (int z = 0; z < m; z++) {
                    C_data[c_index] += A_data[y * m + z] * B_data[x * m + z];
                }
            }
        }
    } else if (A->precision == FLOAT) {
        float *A_data = A->data;
        float *B_data = B_copy->data;
        float *C_data = C->data;

        for (int y = 0; y < l; y++) {
            for (int x = 0; x < n; x++) {
                int c_index = y * n + x;
                C_data[c_index] = 0.0;
                for (int z = 0; z < m; z++) {
                    C_data[c_index] += A_data[y * m + z] * B_data[x * m + z];
                }
            }
        }
    } else if (A->precision == INT) {
        int *A_data = A->data;
        int *B_data = B_copy->data;
        int *C_data = C->data;

        for (int y = 0; y < l; y++) {
            for (int x = 0; x < n; x++) {
                int c_index = y * n + x;
                C_data[c_index] = 0;
                for (int z = 0; z < m; z++) {
                    C_data[c_index] += A_data[y * m + z] * B_data[x * m + z];
                }
            }
        }
    }
}
