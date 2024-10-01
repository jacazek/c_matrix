cmake_minimum_required(VERSION 3.26)
project(matrix CUDA C)
enable_testing()
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "89")

message(STATUS $ENV{LD_LIBRARY_PATH})
# enable SIMD instructions for CPU
add_compile_options(-mavx2 -mavxifma -mavxvnni -march=znver3 -mtune=znver3)

include(ExternalProject)

# define the external dependency location
set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)
#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${EXTERNAL_INSTALL_LOCATION}/lib64)

# Add unity test framework as an external project
ExternalProject_Add(unity
        # get unity from github
        GIT_REPOSITORY https://github.com/ThrowTheSwitch/Unity.git
        CMAKE_ARGS
        # tell cmake to install unity in the external dependency directory
        -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
        # compile unity with test case and double precision support
        # currently not working.... need to figure it out
        -DCMAKE_C_FLAGS="-DUNITY_INCLUDE_DOUBLE -DUNITY_DOUBLE_PRECISION=1e-12f"
        # specifying the expected output causes the external project to be built before satisfying the dependency
        BUILD_BYPRODUCTS ${EXTERNAL_INSTALL_LOCATION}/lib64/libunity.a
)

##### Common library and include locations
# include external headers and binary directories for built external dependencies
include_directories(${EXTERNAL_INSTALL_LOCATION}/include)
link_directories(${EXTERNAL_INSTALL_LOCATION}/lib64)


##### CUDA matrix multiply static library (archive)
add_library(gpu_matrix_multiply gpu_matrix_multiply.cu)
target_link_libraries(gpu_matrix_multiply PRIVATE cuda cudart)

##### AVX matrix multiply static library (archive)
add_library(avx_matrix_multiply avx_matrix_multiply.c)

##### BLOCK matrix multiply static library (archive)
add_library(block_matrix_multiply block_matrix_multiply.c)

##### NAIVE matrix multiply static library (archive)
add_library(naive_matrix_multiply naive_matrix_multiply.c)

##### Matrix main executable
# register main.c as an executable to compile
add_executable(matrix main.c
        matrix.c)
add_dependencies(matrix
        naive_matrix_multiply
        block_matrix_multiply
        avx_matrix_multiply
        gpu_matrix_multiply)
target_link_libraries(matrix
        naive_matrix_multiply
        block_matrix_multiply
        avx_matrix_multiply
        gpu_matrix_multiply
        cudart)
# enable cuda support
set_target_properties(matrix PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON  # For separable compilation (optional)
        BUILD_RPATH "/usr/local/cuda/lib64"
)


##### Matrix test exectutable #####
# register the matrix_test.c as an executable to compile
add_executable(matrix_test matrix_test.c matrix.c)
# tell cmake matrix_test executable has a dependency on the unity external project
add_dependencies(matrix_test
        naive_matrix_multiply
        block_matrix_multiply
        avx_matrix_multiply
        gpu_matrix_multiply
        unity)
target_include_directories(matrix_test PUBLIC ${EXTERNAL_INSTALL_LOCATION}/include)
target_link_libraries(matrix_test
        naive_matrix_multiply
        block_matrix_multiply
        avx_matrix_multiply
        gpu_matrix_multiply
        cudart
        ${EXTERNAL_INSTALL_LOCATION}/lib64/libunity.a)

# also tell matrix_test to compile with the test case and double precision support
target_compile_definitions(matrix_test PUBLIC -DUNITY_SUPPORT_TEST_CASES -DUNITY_SUPPORT_VARIADIC_MACROS -DUNITY_INCLUDE_DOUBLE -DUNITY_DOUBLE_PRECISION=1e-12f)

# enable cuda support
set_target_properties(matrix_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON  # For separable compilation (optional)
        BUILD_RPATH "/usr/local/cuda/lib64"
)
