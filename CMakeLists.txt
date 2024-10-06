cmake_minimum_required(VERSION 3.26)
project(matrix CUDA C)
enable_testing()
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "70;89")

option(CUDA_SUPPORT "Enable CUDA support" ON)
option(AVX_SUPPORT "Enable AVX support" ON)

message(STATUS $ENV{LD_LIBRARY_PATH})
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


if (CUDA_SUPPORT)
    message(STATUS "CUDA SUPPORT ENABLED")
    # provide flag to code enabling GPU support
    add_definitions(-DCUDA_SUPPORT)
    ##### CUDA matrix multiply static library (archive)
    add_library(gpu_matrix_multiply gpu_matrix_multiply.cu)
    target_link_libraries(gpu_matrix_multiply PRIVATE cuda cudart)
else ()
    message(STATUS "CUDA SUPPORT DISABLED")
endif ()

if (AVX_SUPPORT)
    message(STATUS "AVX SUPPORT ENABLED")
    # enable SIMD instructions for CPU
    add_compile_options(-mavx2 -mavxifma -mavxvnni -march=znver3 -mtune=znver3)
    # provide flag to code enabling avx support
    add_definitions(-DAVX_SUPPORT)
    ##### AVX matrix multiply static library (archive)
    add_library(avx_matrix_multiply avx_matrix_multiply.c)
else ()
    message(STATUS "AVX SUPPORT DISABLED")
endif ()

##### BLOCK matrix multiply static library (archive)
add_library(block_matrix_multiply block_matrix_multiply.c)

##### NAIVE matrix multiply static library (archive)
add_library(naive_matrix_multiply naive_matrix_multiply.c)

##### Matrix main executable
# register main.c as an executable to compile
add_executable(matrix main.c
        matrix.c
        naive_memory_aligned_matrix_multiply.c
        naive_memory_aligned_matrix_multiply.h)
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    # does WONDERS for avx (75% reduction in run time)
    # completely destroys block multiplication
    target_compile_options(matrix PUBLIC -O2)
endif ()
add_dependencies(matrix
        naive_matrix_multiply
        block_matrix_multiply)
target_link_libraries(matrix
        naive_matrix_multiply
        block_matrix_multiply)
if (CUDA_SUPPORT)
    add_dependencies(matrix
            gpu_matrix_multiply)
    target_link_libraries(matrix
            gpu_matrix_multiply
            cudart)
    # enable cuda support
    set_target_properties(matrix PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON  # For separable compilation (optional)
            BUILD_RPATH "/usr/local/cuda/lib64"
    )
endif ()

if (AVX_SUPPORT)
    add_dependencies(matrix
            avx_matrix_multiply)
    target_link_libraries(matrix
            avx_matrix_multiply)
endif ()


##### Matrix test exectutable #####
# register the matrix_test.c as an executable to compile
add_executable(matrix_test matrix_test.c matrix.c
        matrix_test.h
        gpu_matrix_multiply_tests.c
        gpu_matrix_multiply_tests.h
        avx_matrix_multiply_tests.c
        avx_matrix_multiply_tests.h
        naive_memory_aligned_matrix_multiply.c
        naive_memory_aligned_matrix_multiply.h)
# tell cmake matrix_test executable has a dependency on the unity external project
add_dependencies(matrix_test
        naive_matrix_multiply
        block_matrix_multiply
        unity)
target_include_directories(matrix_test PUBLIC ${EXTERNAL_INSTALL_LOCATION}/include)
target_link_libraries(matrix_test
        naive_matrix_multiply
        block_matrix_multiply
        ${EXTERNAL_INSTALL_LOCATION}/lib64/libunity.a)

if (CUDA_SUPPORT)
    add_definitions(-DCUDA_SUPPORT)
    add_dependencies(matrix_test
            gpu_matrix_multiply)
    target_link_libraries(matrix_test
            gpu_matrix_multiply
            cudart)
    # enable cuda support
    set_target_properties(matrix_test PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON  # For separable compilation (optional)
            BUILD_RPATH "/usr/local/cuda/lib64"
    )
endif ()

if (AVX_SUPPORT)
    add_dependencies(matrix_test
            avx_matrix_multiply)
    target_link_libraries(matrix_test
            avx_matrix_multiply)
endif ()

# also tell matrix_test to compile with the test case and double precision support
target_compile_definitions(matrix_test PUBLIC -DUNITY_INCLUDE_DOUBLE -DUNITY_DOUBLE_PRECISION=1e-12f)
