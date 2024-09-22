//
// Created by jacob on 9/22/24.
//
#include "stdio.h"
#include "unity/unity.h"
#include "matrix.h"

double A_data[] = {0.38300,-0.35100,-0.46000,0.78200,-0.97800,-0.77100,-0.68500,-0.13800,-0.11400,0.42100,0.42600,0.53000,0.05800,0.37300,-0.63000,0.17000,-0.22300,-0.63800,0.17200,-0.13800,0.06900,-0.57900,-0.58700,-0.00400,-0.08500,-0.97300,0.73600,0.12300,-0.83300,-0.08100,0.52600,0.28100,0.79300,-0.31000,0.21100,-0.93300,0.39300,0.78400,-0.90900,-0.69500,-0.66500,-0.94100,0.36800,0.13500,-0.54400,-0.46300,-0.02000,-0.07500,0.38600,0.76300,-0.43300,0.92900,0.01100,0.19800,0.95600,0.08400,-0.50800,0.92600,-0.57100,0.80200,-0.95800,-0.67600,0.87300,-0.67300};
double B_data[] = {-0.66400,0.58200,0.08700,-0.24600,-0.77400,0.46700,0.30100,-0.56000,-0.49500,-0.45500,-0.19200,-0.60100,-0.41400,-0.39900,-0.72000,-0.27100,-0.15400,-0.18600,0.27600,0.93200,-0.90600,-0.90300,-0.71400,-0.96900,0.72900,0.36700,0.17800,0.06000,0.53900,-0.09800,0.44100,-0.88300,0.31300,0.43400,0.78800,0.67600,-0.20500,0.31700,0.86500,-0.90300,0.85700,-0.63600,0.58400,0.36800,-0.43000,-0.50800,0.68900,0.77100,-0.87600,-0.95700,0.40300,0.73900,0.43400,-0.34800,-0.55600,-0.51900,0.89500,0.75000,-0.34900,-0.98800,-0.62200,-0.24400,-0.38100,-0.32500};
double C_data[] = {0.07004,1.37311,-1.33587,-1.57979,1.00770,1.01135,0.09755,0.32493,1.22991,0.37573,0.07148,-0.25320,-0.73778,-0.65117,0.18481,-0.42411,0.37282,1.03482,-0.39293,-0.00595,0.18907,0.52989,0.19722,0.05020,-0.02503,-0.30115,-0.18535,0.81718,0.12719,-0.80266,-0.97216,-0.18508,-0.11660,0.24152,0.66444,0.70099,-1.55940,0.60714,1.55011,1.20654,0.33243,0.04348,-0.43253,0.60140,0.99222,-0.19261,-0.47573,0.32063,-0.47921,-0.67408,0.41319,-0.19754,0.55364,-0.27450,-0.13193,-1.20440,-1.69474,-1.64249,-0.79982,-0.50191,2.24399,-0.26956,-1.58166,-0.01182};
int A_data_int[] = {383,-351,-460,782,-978,-771,-685,-138,-114,421,426,530,58,373,-630,170,-223,-638,172,-138,69,-579,-587,-4,-85,-973,736,123,-833,-81,526,281,793,-310,211,-933,393,784,-909,-695,-665,-941,368,135,-544,-463,-20,-75,386,763,-433,929,11,198,956,84,-508,926,-571,802,-958,-676,873,-673};
int B_data_int[] = {-664,582,87,-246,-774,467,301,-560,-495,-455,-192,-601,-414,-399,-720,-271,-154,-186,276,932,-906,-903,-714,-969,729,367,178,60,539,-98,441,-883,313,434,788,676,-205,317,865,-903,857,-636,584,368,-430,-508,689,771,-876,-957,403,739,434,-348,-556,-519,895,750,-349,-988,-622,-244,-381,-325};
int C_data_int[] = {70040,1373114,-1335872,-1579794,1007696,1011348,97554,324933,1229912,375725,71482,-253203,-737784,-651173,184809,-424106,372818,1034815,-392926,-5949,189075,529886,197221,50201,-25029,-301148,-185348,817185,127190,-802655,-972157,-185081,-116597,241520,664441,700993,-1559396,607143,1550106,1206541,332430,43484,-432534,601399,992221,-192614,-475734,320628,-479213,-674078,413192,-197543,553644,-274499,-131926,-1204402,-1694735,-1642493,-799820,-501915,2243990,-269563,-1581661,-11817};

int l = 8;
int m = 8;
int n = 8;
MatrixPrecision precision = INT;
matrix_2d *A;
matrix_2d *B;
matrix_2d *C;

MatrixPrecision double_precision = DOUBLE;
matrix_2d *A_double;
matrix_2d *B_double;
matrix_2d *C_double;


void setUp() {
    A = matrix2D_new(precision, m, l);
    B = matrix2D_new(precision, n, m);
    C = matrix2D_new(precision, B->x_length, A->y_length);

    matrix2D_fill(A, A_data_int);
    matrix2D_fill(B, B_data_int);

    A_double = matrix2D_new(double_precision, m, l);
    B_double = matrix2D_new(double_precision, n, m);
    C_double = matrix2D_new(double_precision, B->x_length, A->y_length);

    matrix2D_fill(A_double, A_data);
    matrix2D_fill(B_double, B_data);
}

void tearDown() {

}

void naive_matmul_works_correctly() {
    matrix_matmul(A, B, C, NAIVE);

    TEST_ASSERT_EQUAL_INT_ARRAY(C_data_int, C->data, C->y_length * C->x_length);
}

//void naive_matmul_double_works_correctly() {
//    matrix_matmul(A_double, B_double, C_double, NAIVE);
//
//    TEST_ASSERT_EQUAL_DOUBLE_ARRAY(C_data, C_double->data, C_double->y_length * C_double->x_length);
//}

void block_matmul_works_correctly() {
    matrix_matmul(A, B, C, BLOCK);

    TEST_ASSERT_EQUAL_INT_ARRAY(C_data_int, C->data, C->y_length * C->x_length);
}

//void simd_double_matmul_works_correctly() {
//    matrix_matmul(A, B, C, AVX);
//    int data[] = {-17221, -468072, 696172, 795056, 232591, 653281, -392070, -349754};
//    TEST_ASSERT_EQUAL_INT_ARRAY(data, C->data, C->y_length * C->x_length);
//}

void simd_int_matmul_works_correctly() {
    matrix_matmul(A, B, C, AVX);
//    int data[] = {-17221, -468072, 696172, 795056, 232591, 653281, -392070, -349754};
    TEST_ASSERT_EQUAL_INT_ARRAY(C_data_int, C->data, C->y_length * C->x_length);
}

void transpose_works() {
    int size = 3;
    int* data_to_transpose = malloc(sizeof(int) * size * size);
    for (int i = 0; i <9; i++) {
        data_to_transpose[i] = i+1;
    }
    int expected_output[] = {1, 4, 7, 2, 5, 8, 3, 6, 9};

    matrix_2d * matrix_to_transpose = matrix2D_new(INT, size, size);
    matrix2D_fill(matrix_to_transpose, data_to_transpose);

    matrix2D_transpose(matrix_to_transpose);
    TEST_ASSERT_EQUAL_INT_ARRAY(expected_output, matrix_to_transpose->data, matrix_to_transpose->x_length * matrix_to_transpose->y_length);
}

int main() {
    UNITY_BEGIN();

    RUN_TEST(naive_matmul_works_correctly);
//    RUN_TEST(naive_matmul_double_works_correctly);
    RUN_TEST(block_matmul_works_correctly);
//    RUN_TEST(simd_double_matmul_works_correctly);
    RUN_TEST(simd_int_matmul_works_correctly);
    RUN_TEST(transpose_works);
    return UNITY_END();
}