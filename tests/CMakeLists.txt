set(JACOBI_CLI_PARSER_TEST_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/cli_parser_tests.cu"
    ${JACOBI_INTERFACE_SOURCES}
)

if(JACOBI_USE_CMAKE_CUDA_LANGUAGE)
    add_executable(jacobi-cli-parser-tests ${JACOBI_CLI_PARSER_TEST_SOURCES} ${JACOBI_HEADERS})
    target_include_directories(jacobi-cli-parser-tests PRIVATE "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}")
    set_target_properties(jacobi-cli-parser-tests PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
        CUDA_SEPARABLE_COMPILATION ON
    )
    set(JACOBI_CLI_PARSER_TEST_COMMAND jacobi-cli-parser-tests)
else()
    if(WIN32)
        set(JACOBI_CLI_PARSER_TEST_EXE "${JACOBI_OUTPUT_DIR}/jacobi-cli-parser-tests.exe")
    else()
        set(JACOBI_CLI_PARSER_TEST_EXE "${JACOBI_OUTPUT_DIR}/jacobi-cli-parser-tests")
    endif()

    set(JACOBI_CLI_PARSER_TEST_STAMP "${CMAKE_CURRENT_BINARY_DIR}/jacobi-cli-parser-tests.stamp")

    add_custom_command(
        OUTPUT "${JACOBI_CLI_PARSER_TEST_STAMP}"
        BYPRODUCTS "${JACOBI_CLI_PARSER_TEST_EXE}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${JACOBI_OUTPUT_DIR}"
        COMMAND "${JACOBI_NVCC_EXECUTABLE}" ${JACOBI_NVCC_ARGS}
                ${JACOBI_CLI_PARSER_TEST_SOURCES}
                -o "${JACOBI_CLI_PARSER_TEST_EXE}"
        COMMAND "${CMAKE_COMMAND}" -E touch "${JACOBI_CLI_PARSER_TEST_STAMP}"
        DEPENDS ${JACOBI_CLI_PARSER_TEST_SOURCES} ${JACOBI_HEADERS}
        VERBATIM
        COMMENT "Building jacobi-cli-parser-tests via nvcc driver mode"
    )

    add_custom_target(jacobi-cli-parser-tests DEPENDS "${JACOBI_CLI_PARSER_TEST_STAMP}")
    set(JACOBI_CLI_PARSER_TEST_COMMAND "${JACOBI_CLI_PARSER_TEST_EXE}")
endif()

add_test(NAME jacobi-cli-parser-tests COMMAND "${JACOBI_CLI_PARSER_TEST_COMMAND}")
set_tests_properties(jacobi-cli-parser-tests PROPERTIES LABELS unit)

add_custom_target(check
    COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure
    DEPENDS jacobi-cli-parser-tests
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    COMMENT "Running project test suite"
)
