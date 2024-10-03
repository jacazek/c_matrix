//
// Created by jacob on 10/2/24.
//

#include "avx_matrix_multiply_tests.h"

#include "avx_matrix_multiply.h"
#include "matrix_test.h"

void simd_int_matmul_works_correctly() {
    avx_matmul(A, B, C);
    //    int data[] = {-17221, -468072, 696172, 795056, 232591, 653281, -392070, -349754};
    TEST_ASSERT_EQUAL_INT_ARRAY(C_data_int, C->data, C->y_length * C->x_length);
}

void simd_matmul_double_works_correctly() {
    avx_matmul(A_double, B_double, C_double);
    TEST_ASSERT_DOUBLE_ARRAY_WITHIN(.0001, C_data, C_double->data, C_double->y_length * C_double->x_length);
}

void simd_matmul_float_works_correctly() {
    avx_matmul(A_float, B_float, C_float);
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(.0001, C_data_float, C_float->data, C_float->y_length * C_float->x_length);
}

void simd_matmul_int_works_on_matrices_not_multiples_of_8() {
    int size = 10;
    MatrixPrecision rect_precision = INT;
    matrix_2d *A_rect = matrix2D_new(rect_precision, size, size);
    matrix_2d *B_rect = matrix2D_new(rect_precision, size, size);
    matrix_2d *C_rect = matrix2D_new(rect_precision, size, size);

    int numberOfElements = size * size;
    int _a_data[numberOfElements];
    int _b_data[numberOfElements];
    int _c_expected[numberOfElements];

    for (int i = 0; i < numberOfElements; i++) {
        _a_data[i] = 1;
        _b_data[i] = 2;
        _c_expected[i] = 2 * size;
    }

    matrix2D_fill(A_rect, _a_data);
    matrix2D_fill(B_rect, _b_data);

    avx_matmul(A_rect, B_rect, C_rect);

    TEST_ASSERT_EQUAL_INT_ARRAY(_c_expected, C_rect->data, numberOfElements);
    matrix2D_destroy(&A_rect);
    matrix2D_destroy(&B_rect);
    matrix2D_destroy(&C_rect);
}

void simd_matmul_float_works_on_matrices_not_multiples_of_8() {
    int size = 10;
    MatrixPrecision rect_precision = FLOAT;
    matrix_2d *A_rect = matrix2D_new(rect_precision, size, size);
    matrix_2d *B_rect = matrix2D_new(rect_precision, size, size);
    matrix_2d *C_rect = matrix2D_new(rect_precision, size, size);

    int numberOfElements = size * size;
    float _a_data[numberOfElements];
    float _b_data[numberOfElements];
    float _c_expected[numberOfElements];

    for (int i = 0; i < numberOfElements; i++) {
        _a_data[i] = 1;
        _b_data[i] = 2;
        _c_expected[i] = 2 * size;
    }

    matrix2D_fill(A_rect, _a_data);
    matrix2D_fill(B_rect, _b_data);

    avx_matmul(A_rect, B_rect, C_rect);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(_c_expected, C_rect->data, numberOfElements);
    matrix2D_destroy(&A_rect);
    matrix2D_destroy(&B_rect);
    matrix2D_destroy(&C_rect);
}

void simd_matmul_double_works_on_matrices_not_multiples_of_8() {
    int size = 10;
    MatrixPrecision rect_precision = DOUBLE;
    matrix_2d *A_rect = matrix2D_new(rect_precision, size, size);
    matrix_2d *B_rect = matrix2D_new(rect_precision, size, size);
    matrix_2d *C_rect = matrix2D_new(rect_precision, size, size);

    int numberOfElements = size * size;
    double _a_data[numberOfElements];
    double _b_data[numberOfElements];
    double _c_expected[numberOfElements];

    for (int i = 0; i < numberOfElements; i++) {
        _a_data[i] = 1;
        _b_data[i] = 2;
        _c_expected[i] = 2 * size;
    }

    matrix2D_fill(A_rect, _a_data);
    matrix2D_fill(B_rect, _b_data);

    avx_matmul(A_rect, B_rect, C_rect);

    TEST_ASSERT_EQUAL_DOUBLE_ARRAY(_c_expected, C_rect->data, numberOfElements);
    matrix2D_destroy(&A_rect);
    matrix2D_destroy(&B_rect);
    matrix2D_destroy(&C_rect);
}


void register_avx_matrix_multiply_tests() {
    RUN_TEST(simd_matmul_double_works_correctly);
    RUN_TEST(simd_int_matmul_works_correctly);
    RUN_TEST(simd_matmul_float_works_correctly);
    RUN_TEST(simd_matmul_int_works_on_matrices_not_multiples_of_8);
    RUN_TEST(simd_matmul_float_works_on_matrices_not_multiples_of_8);
    RUN_TEST(simd_matmul_double_works_on_matrices_not_multiples_of_8);
}
