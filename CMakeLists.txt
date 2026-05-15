cmake_minimum_required(VERSION 3.24)

project(jacobi_svd_cuda LANGUAGES NONE)

set(JACOBI_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/build")

option(JACOBI_USE_CMAKE_CUDA_LANGUAGE
       "Use CMake native CUDA language mode (requires VS CUDA toolset on Windows)."
       OFF)
option(JACOBI_BUILD_TESTS "Build the project test suite." OFF)

add_subdirectory(src)

if(JACOBI_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
