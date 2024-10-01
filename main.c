#include <stdio.h>

#include "matrix.h"
#include "time.h"
#include "pthread.h"

struct thread_data {
    matrix_2d *A;
    matrix_2d *B;
};

void print_matrix(char *name, matrix_2d *matrix) {
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
    } else if (matrix->precision == FLOAT) {
        for (int i = 0; i < matrix->y_length; i++) {
            for (int j = 0; j < matrix->x_length; j++) {
                float value;
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

void *run_naive_test(void *data) {
    struct thread_data *thread_data = data;
    matrix_2d *C = matrix2D_new(thread_data->A->precision, thread_data->B->x_length, thread_data->A->y_length);
    printf("NAIVE matmul starting...\n");
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    matrix_matmul(thread_data->A, thread_data->B, C, NAIVE);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec) +
                        (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("NAIVE elapsed time: %f seconds\n", time_taken);
    matrix2D_destroy(&C);
    return NULL;
}

void *run_block_test(void *data) {
    struct thread_data *thread_data = data;
    matrix_2d *C = matrix2D_new(thread_data->A->precision, thread_data->B->x_length, thread_data->A->y_length);
    printf("BLOCK matmul starting...\n");
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    matrix_matmul(thread_data->A, thread_data->B, C, BLOCK);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec) +
                        (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("BLOCK elapsed time: %f seconds\n", time_taken);
    matrix2D_destroy(&C);
    return NULL;
}

void *run_avx_test(void *data) {
    struct thread_data *thread_data = data;
    matrix_2d *B_clone = matrix2D_copy(thread_data->B);
    matrix_2d *C = matrix2D_new(thread_data->A->precision, thread_data->B->x_length, thread_data->A->y_length);
    printf("AVX matmul starting...\n");
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    matrix_matmul(thread_data->A, B_clone, C, AVX);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec) +
                        (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("AVX elapsed time: %f seconds\n", time_taken);
    matrix2D_destroy(&B_clone);
    matrix2D_destroy(&C);
    return NULL;
}

void *run_gpu_test(void *data) {
    struct thread_data *thread_data = data;
    matrix_2d *C = matrix2D_new(thread_data->A->precision, thread_data->B->x_length, thread_data->A->y_length);
    printf("GPU matmul starting...\n");
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    matrix_matmul(thread_data->A, thread_data->B, C, GPU);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec) +
                        (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("GPU elapsed time: %f seconds\n", time_taken);
    matrix2D_destroy(&C);
    return NULL;
}


int main() {
    clock_t start, end;
    double cpu_time_used;
    int l = 2048;
    int m = 2048;
    int n = 2048;
    int size = 3;
    MatrixPrecision precision = DOUBLE;
    matrix_2d *A = matrix2D_new(precision, m, l);
    matrix_2d *B = matrix2D_new(precision, n, m);
    // matrix_2d *C = matrix2D_new(precision, B->x_length, A->y_length);

    srand(1);
    matrix_random(A);
    matrix_random(B);

    struct thread_data thread_data = {
        .A = A,
        .B = B,
    };
    pthread_t threads[4];
    pthread_create(&threads[0], NULL, run_naive_test, &thread_data);
    pthread_create(&threads[1], NULL, run_block_test, &thread_data);
    pthread_create(&threads[2], NULL, run_avx_test, &thread_data);
    pthread_create(&threads[3], NULL, run_gpu_test, &thread_data);
    for (int i = 0; i < 4; i++) {
        pthread_join(threads[i], NULL);
    }
    //
    // printf("NAIVE matmul starting...\n");
    // start = clock();
    // matrix_matmul(A, B, C, NAIVE);
    // end = clock();
    // cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    // printf("NAIVE elapsed time: %f seconds\n", cpu_time_used);
    // //
    // printf("BLOCK matmul starting...\n");
    // start = clock();
    // matrix_matmul(A, B, C, BLOCK);
    // end = clock();
    // cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    // printf("BLOCK elapsed time: %f seconds\n", cpu_time_used);
    // //
    // printf("AVX matmul starting...\n");
    // start = clock();
    // matrix_matmul(A, B, C, AVX);
    // end = clock();
    // cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    // printf("AVX elapsed time: %f seconds\n", cpu_time_used);
    //
    // printf("GPU matmul starting...\n");
    // start = clock();
    // matrix_matmul(A, B, C, GPU);
    // end = clock();
    // cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    // printf("GPU elapsed time: %f seconds\n", cpu_time_used);

    // double outValue = 0;
    // matrix2D_get_element(C, 0, 0, &outValue);

    // printf("Hello, World! %0.6f\n", outValue);
    // print_matrix("A", A);
    // print_matrix("B", B);
    // print_matrix("C", C);

    matrix2D_destroy(&A);
    matrix2D_destroy(&B);
    // matrix2D_destroy(&C);
    return 0;
}
