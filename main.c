#include <stdio.h>

#include "matrix.h"
#include "time.h"

void print_matrix(char* name, matrix_2d* matrix) {
    printf("%s\n", name);
    printf("[");
    if (matrix->precision == DOUBLE) {
        for (int i = 0; i < matrix->y_length; i++) {
            for (int j = 0; j < matrix->x_length; j++) {
                double value;
                matrix2D_get_element(matrix, j, i, &value);
                printf("%.5f%s", value, j < matrix->x_length - 1 ? "," : "");
            }
            if (i < matrix->x_length - 1) printf(";");
        }
    } else if (matrix->precision == INT) {
        for (int i = 0; i < matrix->y_length; i++) {
            for (int j = 0; j < matrix->x_length; j++) {
                int value;
                matrix2D_get_element(matrix, j, i, &value);
                printf("%i%s", value, j < matrix->x_length - 1 ? "," : "");
            }
            if (i < matrix->x_length - 1) printf(";");
        }
    }
    printf("]\n");
}

int main() {
    clock_t start, end;
    double cpu_time_used;
    int l = 2048;
    int m = 2048;
    int n = 2048;
    int size = 3;
    MatrixPrecision precision = DOUBLE;
    matrix_2d * A = matrix2D_new(precision, m, l);
    matrix_2d * B = matrix2D_new(precision, n, m);
    matrix_2d * C = matrix2D_new(precision, B->x_length, A->y_length);

    srand(1);
    matrix_random(A);
    matrix_random(B);

    printf("NAIVE matmul starting...\n");
    start = clock();
    matrix_matmul(A, B, C, NAIVE);
    end  = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("NAIVE elapsed time: %f seconds\n", cpu_time_used);

    printf("BLOCK matmul starting...\n");
    start = clock();
    matrix_matmul(A, B, C, BLOCK);
    end  = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("BLOCK elapsed time: %f seconds\n", cpu_time_used);

    printf("AVX matmul starting...\n");
    start = clock();
    matrix_matmul(A, B, C, AVX);
    end  = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("AVX elapsed time: %f seconds\n", cpu_time_used);

    printf("GPU matmul starting...\n");
    start = clock();
    matrix_matmul(A, B, C, GPU);
    end  = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("GPU elapsed time: %f seconds\n", cpu_time_used);

    // double outValue = 0;
    // matrix2D_get_element(C, 0, 0, &outValue);

    // printf("Hello, World! %0.6f\n", outValue);
//    print_matrix("A", A);
//    print_matrix("B", B);
//    print_matrix("C", C);

    matrix2D_destroy(&A);
    matrix2D_destroy(&B);
    matrix2D_destroy(&C);
    return 0;
}
