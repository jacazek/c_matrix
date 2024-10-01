//
// Created by jacob on 9/23/24.
//

#include "naive_matrix_multiply.h"

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
    } else if (A->precision == FLOAT) {
        float *A_data = A->data;
        float *B_data = B->data;
        float *C_data = C->data;

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
