cmake_minimum_required(VERSION 3.26)
project(matrix C)
enable_testing()
set(CMAKE_C_STANDARD 17)

#include(CheckCCompilerFlag)
#check_c_compiler_flag("-mavx" COMPILER_SUPPORTS_AVX)
#message(STATUS {COMPILER_SUPPORTS_AVX})

# enable SIMD instructions for CPU
add_compile_options(-mavx2 -mavxifma -mavxvnni -march=znver3 -mtune=znver3)

include(ExternalProject)

set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)
ExternalProject_Add(unity
        GIT_REPOSITORY https://github.com/ThrowTheSwitch/Unity.git
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
            -DCMAKE_C_FLAGS="-DUNITY_SUPPORT_TEST_CASES -DUNITY_SUPPORT_VARIADIC_MACROS -DUNITY_INCLUDE_DOUBLE -DUNITY_DOUBLE_PRECISION=1e-12f"
        # specifying the expected output causes the external project to be built before satisfying the dependency
        BUILD_BYPRODUCTS ${EXTERNAL_INSTALL_LOCATION}/lib64/libunity.a
)
#target_compile_definitions(unity PUBLIC -DUNITY_SUPPORT_TEST_CASES)
include_directories(${EXTERNAL_INSTALL_LOCATION}/include)
link_directories(${EXTERNAL_INSTALL_LOCATION}/lib64)

#ExternalProject_Get_Property(unity install_dir)
#ExternalProject_Get_Property(unity binary_dir)
#message("${binary_dir}")

add_executable(matrix main.c
        matrix.c)

add_executable(matrix_test matrix_test.c matrix.c)
add_dependencies(matrix_test unity)
target_include_directories(matrix_test PUBLIC ${EXTERNAL_INSTALL_LOCATION}/include)
target_link_libraries(matrix_test ${EXTERNAL_INSTALL_LOCATION}/lib64/libunity.a)
target_compile_definitions(matrix_test PUBLIC -DUNITY_SUPPORT_TEST_CASES -DUNITY_SUPPORT_VARIADIC_MACROS -DUNITY_INCLUDE_DOUBLE -DUNITY_DOUBLE_PRECISION=1e-12f)
#add_test(matrix_test_test matrix_test)

