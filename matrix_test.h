//
// Created by jacob on 10/2/24.
//

#ifndef MATRIX_TEST_H
#define MATRIX_TEST_H
#include "unity/unity.h"
#include "matrix.h"
extern double A_data[];
extern double B_data[];
extern double C_data[];

extern float A_data_float[];
extern float B_data_float[];
extern float C_data_float[];

extern int A_data_int[];
extern int B_data_int[];
extern int C_data_int[];

extern int l;
extern int m;
extern int n;
extern MatrixPrecision precision;
extern matrix_2d *A;
extern matrix_2d *B;
extern matrix_2d *C;

extern MatrixPrecision double_precision;
extern matrix_2d *A_double;
extern matrix_2d *B_double;
extern matrix_2d *C_double;

extern MatrixPrecision single_precision;
extern matrix_2d *A_float;
extern matrix_2d *B_float;
extern matrix_2d *C_float;
#endif //MATRIX_TEST_H
