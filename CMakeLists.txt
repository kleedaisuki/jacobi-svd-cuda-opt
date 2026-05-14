cmake_minimum_required(VERSION 3.24)

project(jacobi_svd_cuda LANGUAGES NONE)

set(JACOBI_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/build")

option(JACOBI_USE_CMAKE_CUDA_LANGUAGE
       "Use CMake native CUDA language mode (requires VS CUDA toolset on Windows)."
       OFF)

set(JACOBI_INTERFACE_SOURCES
    "${CMAKE_SOURCE_DIR}/src/interfaces/cli_parser.cu"
    "${CMAKE_SOURCE_DIR}/src/interfaces/cli_support.cu"
    "${CMAKE_SOURCE_DIR}/src/interfaces/main.cu"
)

set(JACOBI_APPLICATION_SOURCES
    "${CMAKE_SOURCE_DIR}/src/application/pipeline.cu"
    "${CMAKE_SOURCE_DIR}/src/application/pipeline_helpers.cu"
)

set(JACOBI_DOMAIN_SOURCES
    "${CMAKE_SOURCE_DIR}/src/domain/cuda_error.cu"
    "${CMAKE_SOURCE_DIR}/src/domain/device_matrix.cu"
    "${CMAKE_SOURCE_DIR}/src/domain/jacobi_svd.cu"
    "${CMAKE_SOURCE_DIR}/src/domain/layout_transpose_autotune.cu"
)

set(JACOBI_INFRASTRUCTURE_SOURCES
    "${CMAKE_SOURCE_DIR}/src/infrastructure/files.cu"
    "${CMAKE_SOURCE_DIR}/src/infrastructure/mat_dispatch_reader.cu"
    "${CMAKE_SOURCE_DIR}/src/infrastructure/mat_file.cu"
    "${CMAKE_SOURCE_DIR}/src/infrastructure/pinned_host_task_buffer.cu"
    "${CMAKE_SOURCE_DIR}/src/infrastructure/txt_file.cu"
)

set(JACOBI_HEADERS
    "${CMAKE_SOURCE_DIR}/include/jacobi/svd/application/pipeline.hpp"
    "${CMAKE_SOURCE_DIR}/include/jacobi/svd/domain/cuda_error.hpp"
    "${CMAKE_SOURCE_DIR}/include/jacobi/svd/domain/device_matrix.hpp"
    "${CMAKE_SOURCE_DIR}/include/jacobi/svd/domain/jacobi_svd.hpp"
    "${CMAKE_SOURCE_DIR}/include/jacobi/svd/domain/jacobi_svd_config.hpp"
    "${CMAKE_SOURCE_DIR}/include/jacobi/svd/domain/jacobi_svd_result.hpp"
    "${CMAKE_SOURCE_DIR}/include/jacobi/svd/domain/kernels.hpp"
    "${CMAKE_SOURCE_DIR}/include/jacobi/svd/domain/layout_transpose.hpp"
    "${CMAKE_SOURCE_DIR}/include/jacobi/svd/io/files.hpp"
    "${CMAKE_SOURCE_DIR}/include/jacobi/svd/io/io.hpp"
    "${CMAKE_SOURCE_DIR}/include/jacobi/svd/io/mat_dispatch.hpp"
    "${CMAKE_SOURCE_DIR}/include/jacobi/svd/io/mat_file.hpp"
    "${CMAKE_SOURCE_DIR}/include/jacobi/svd/io/mat_metadata.hpp"
    "${CMAKE_SOURCE_DIR}/include/jacobi/svd/io/matrix.hpp"
    "${CMAKE_SOURCE_DIR}/include/jacobi/svd/io/matrix_stream.hpp"
    "${CMAKE_SOURCE_DIR}/include/jacobi/svd/io/pinned_host_task_buffer.hpp"
    "${CMAKE_SOURCE_DIR}/include/jacobi/svd/io/txt_file.hpp"
    "${CMAKE_SOURCE_DIR}/src/application/kernel_stage.hpp"
    "${CMAKE_SOURCE_DIR}/src/application/output_stage.hpp"
    "${CMAKE_SOURCE_DIR}/src/application/pipeline_detail.hpp"
    "${CMAKE_SOURCE_DIR}/src/application/result_writer.hpp"
    "${CMAKE_SOURCE_DIR}/src/application/text_testcase_source.hpp"
    "${CMAKE_SOURCE_DIR}/src/application/thread_pool.hpp"
    "${CMAKE_SOURCE_DIR}/src/domain/cuda_check.cuh"
    "${CMAKE_SOURCE_DIR}/src/domain/device_buffer.cuh"
    "${CMAKE_SOURCE_DIR}/src/domain/jacobi_rotation_kernels.cuh"
    "${CMAKE_SOURCE_DIR}/src/domain/jacobi_schedule.cuh"
    "${CMAKE_SOURCE_DIR}/src/domain/jacobi_svd_detail.hpp"
    "${CMAKE_SOURCE_DIR}/src/domain/layout_transpose_kernels.cuh"
    "${CMAKE_SOURCE_DIR}/src/domain/matrix_index.cuh"
    "${CMAKE_SOURCE_DIR}/src/infrastructure/append_mapped_output_file.hpp"
    "${CMAKE_SOURCE_DIR}/src/infrastructure/io_detail.hpp"
    "${CMAKE_SOURCE_DIR}/src/infrastructure/memory_mapped_input_file.hpp"
    "${CMAKE_SOURCE_DIR}/src/interfaces/arg_parser.hpp"
    "${CMAKE_SOURCE_DIR}/src/interfaces/cli_actions.hpp"
    "${CMAKE_SOURCE_DIR}/src/interfaces/cli.hpp"
    "${CMAKE_SOURCE_DIR}/src/interfaces/cli_types.hpp"
)

set(JACOBI_SOURCES
    ${JACOBI_INTERFACE_SOURCES}
    ${JACOBI_APPLICATION_SOURCES}
    ${JACOBI_DOMAIN_SOURCES}
    ${JACOBI_INFRASTRUCTURE_SOURCES}
)

if(WIN32)
    set(JACOBI_EXE_PATH "${JACOBI_OUTPUT_DIR}/jacobi-svd-cuda.exe")
else()
    set(JACOBI_EXE_PATH "${JACOBI_OUTPUT_DIR}/jacobi-svd-cuda")
endif()

set(JACOBI_BUILD_STAMP "${CMAKE_CURRENT_BINARY_DIR}/jacobi-svd-cuda.stamp")

if(JACOBI_USE_CMAKE_CUDA_LANGUAGE)
    enable_language(CXX)
    enable_language(CUDA)

    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)

    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${JACOBI_OUTPUT_DIR}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${JACOBI_OUTPUT_DIR}")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${JACOBI_OUTPUT_DIR}")

    foreach(config Debug Release RelWithDebInfo MinSizeRel)
        string(TOUPPER "${config}" config_upper)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config_upper} "${JACOBI_OUTPUT_DIR}")
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${config_upper} "${JACOBI_OUTPUT_DIR}")
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${config_upper} "${JACOBI_OUTPUT_DIR}")
    endforeach()

    add_executable(jacobi-svd-cuda ${JACOBI_SOURCES} ${JACOBI_HEADERS})
    target_include_directories(jacobi-svd-cuda PRIVATE "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}")
    set_target_properties(jacobi-svd-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
    find_program(JACOBI_NVCC_EXECUTABLE
                 NAMES nvcc
                 HINTS "$ENV{CUDA_PATH}/bin"
                 REQUIRED)

    set(JACOBI_NVCC_ARGS
        -std=c++20
        -I"${CMAKE_SOURCE_DIR}/include"
        -I"${CMAKE_SOURCE_DIR}"
    )

    if(WIN32)
        list(APPEND JACOBI_NVCC_ARGS -Xcompiler=/utf-8)
    endif()

    add_custom_command(
        OUTPUT "${JACOBI_BUILD_STAMP}"
        BYPRODUCTS "${JACOBI_EXE_PATH}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${JACOBI_OUTPUT_DIR}"
        COMMAND "${JACOBI_NVCC_EXECUTABLE}" ${JACOBI_NVCC_ARGS} ${JACOBI_SOURCES} -o "${JACOBI_EXE_PATH}"
        COMMAND "${CMAKE_COMMAND}" -E touch "${JACOBI_BUILD_STAMP}"
        DEPENDS ${JACOBI_SOURCES} ${JACOBI_HEADERS}
        VERBATIM
        COMMENT "Building jacobi-svd-cuda via nvcc driver mode"
    )

    add_custom_target(jacobi-svd-cuda ALL DEPENDS "${JACOBI_BUILD_STAMP}")
endif()
