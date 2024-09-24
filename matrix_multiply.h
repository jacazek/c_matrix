//
// Created by jacob on 9/23/24.
//

#ifndef MATRIX_MATRIX_MULTIPLY_H
#define MATRIX_MATRIX_MULTIPLY_H
#include "matrix.h"

// cuda compiles a C++ library.  we need to explicitly make the library available for C
#ifdef __cplusplus
extern "C" {
#endif
      void matrix2D_gpu_matmul(matrix_2d *A, matrix_2d *B, matrix_2d *C);
#ifdef __cplusplus
}
#endif

#endif //MATRIX_MATRIX_MULTIPLY_H
