cmake_minimum_required(VERSION 3.20)

project(
    ultrahigh_ann
    VERSION 0.2.0
    DESCRIPTION "Sublinear coordinate acquisition for nearest-neighbor search"
    LANGUAGES CXX
)

include(GNUInstallDirs)

find_package(OpenMP REQUIRED COMPONENTS CXX)

option(
    ULTRAHIGH_ANN_ENABLE_CUDA
    "Build the optional CUDA backends"
    OFF
)

set(
    ULTRAHIGH_ANN_GPU_SOURCES
    src/core/cuda_dense_l1_scan_index_stub.cpp
    src/core/cuda_dense_l2_scan_index_stub.cpp
    src/core/cuda_runtime_info_stub.cpp
    src/hnsvw25/l1/cuda_importance_sampling_stub.cpp
    src/hnsvw25/l2/cuda_importance_sampling_stub.cpp
    src/core/cuda_float_projection_stub.cpp
    src/hnsvw25/l1/hierarchical/cuda_median_l1_scan_index_stub.cpp
)
if(ULTRAHIGH_ANN_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(
            FATAL_ERROR
            "ULTRAHIGH_ANN_ENABLE_CUDA=ON requires a CUDA compiler"
        )
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(
        ULTRAHIGH_ANN_GPU_SOURCES
        src/core/cuda_dense_l1_scan_index.cu
        src/core/cuda_dense_l2_scan_index.cu
        src/core/cuda_runtime_info.cu
        src/hnsvw25/l1/cuda_importance_sampling.cu
        src/hnsvw25/l2/cuda_importance_sampling.cu
        src/core/cuda_float_projection.cu
        src/hnsvw25/l1/hierarchical/cuda_median_l1_scan_index.cu
    )
endif()

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(ULTRAHIGH_ANN_BUILD_BENCHMARKS_DEFAULT ON)
    set(ULTRAHIGH_ANN_BUILD_TOOLS_DEFAULT ON)
else()
    set(ULTRAHIGH_ANN_BUILD_BENCHMARKS_DEFAULT OFF)
    set(ULTRAHIGH_ANN_BUILD_TOOLS_DEFAULT OFF)
endif()

option(
    ULTRAHIGH_ANN_BUILD_BENCHMARKS
    "Build the unified nearest-neighbor benchmark executable"
    "${ULTRAHIGH_ANN_BUILD_BENCHMARKS_DEFAULT}"
)
unset(ULTRAHIGH_ANN_BUILD_BENCHMARKS_DEFAULT)

option(
    ULTRAHIGH_ANN_BUILD_TOOLS
    "Build command-line preprocessing tools"
    "${ULTRAHIGH_ANN_BUILD_TOOLS_DEFAULT}"
)
unset(ULTRAHIGH_ANN_BUILD_TOOLS_DEFAULT)

add_library(
    ultrahigh_ann
    src/datasets/representative_query_dataset.cpp
    src/exact/l1/cuda_exact_l1_index.cpp
    src/exact/l1/exact_l1_index.cpp
    src/exact/l2/cuda_exact_l2_index.cpp
    src/exact/l2/exact_l2_index.cpp
    src/coordinate_sampling/coordinate_sampling.cpp
    src/coordinate_sampling/cuda_sampled_coordinate_l1_ann_index.cpp
    src/coordinate_sampling/cuda_sampled_coordinate_l2_ann_index.cpp
    src/coordinate_sampling/filter_columns.cpp
    src/coordinate_sampling/l1_coordinate_embedding.cpp
    src/coordinate_sampling/l2_coordinate_embedding.cpp
    src/coordinate_sampling/sampled_coordinate_l1_ann_index.cpp
    src/coordinate_sampling/sampled_coordinate_l2_ann_index.cpp
    src/coordinate_sampling/uniform_l1_ann_index.cpp
    src/coordinate_sampling/uniform_l2_ann_index.cpp
    src/coordinate_sampling/uniform_probabilities.cpp
    src/hnsvw25/l1/flat/cuda_l1_ann_index.cpp
    src/hnsvw25/l1/flat/l1_ann_index.cpp
    src/hnsvw25/l1/hierarchical/cuda_hierarchical_l1_ann_index.cpp
    src/hnsvw25/l1/hierarchical/hierarchical_l1_ann_index.cpp
    src/hnsvw25/l1/hierarchical/l1_hierarchical_projection.cpp
    src/hnsvw25/l1/importance_sampling.cpp
    src/hnsvw25/l2/flat/l2_ann_index.cpp
    src/hnsvw25/l2/flat/cuda_l2_ann_index.cpp
    src/hnsvw25/l2/hierarchical/cuda_hierarchical_l2_ann_index.cpp
    src/hnsvw25/l2/hierarchical/hierarchical_l2_ann_index.cpp
    src/hnsvw25/l2/hierarchical/l2_hierarchical_projection.cpp
    src/hnsvw25/l2/importance_sampling.cpp
    ${ULTRAHIGH_ANN_GPU_SOURCES}
    src/io/importance_probability_io.cpp
    src/io/sha256.cpp
    src/core/dense_l2_scan_index.cpp
    src/core/l1_distance.cpp
    src/core/l2_distance.cpp
)
add_library(ultrahigh_ann::ultrahigh_ann ALIAS ultrahigh_ann)

target_link_libraries(
    ultrahigh_ann
    PRIVATE
        OpenMP::OpenMP_CXX
)

if(ULTRAHIGH_ANN_ENABLE_CUDA)
    target_link_libraries(
        ultrahigh_ann
        PRIVATE
            CUDA::cublas
            CUDA::cudart
    )
    set_target_properties(
        ultrahigh_ann
        PROPERTIES
            CUDA_STANDARD 20
            CUDA_STANDARD_REQUIRED YES
    )
endif()

if(ULTRAHIGH_ANN_BUILD_BENCHMARKS)
    set(ULTRAHIGH_ANN_GIT_COMMIT "unknown")
    set(ULTRAHIGH_ANN_GIT_AVAILABLE 0)
    set(ULTRAHIGH_ANN_GIT_DIRTY 0)
    find_package(Git QUIET)
    if(Git_FOUND)
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            RESULT_VARIABLE ULTRAHIGH_ANN_GIT_RESULT
            OUTPUT_VARIABLE ULTRAHIGH_ANN_GIT_COMMIT
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(ULTRAHIGH_ANN_GIT_RESULT EQUAL 0)
            set(ULTRAHIGH_ANN_GIT_AVAILABLE 1)
            execute_process(
                COMMAND "${GIT_EXECUTABLE}" status --porcelain
                        --untracked-files=normal
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                RESULT_VARIABLE ULTRAHIGH_ANN_GIT_STATUS_RESULT
                OUTPUT_VARIABLE ULTRAHIGH_ANN_GIT_STATUS
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
            if(NOT ULTRAHIGH_ANN_GIT_STATUS_RESULT EQUAL 0 OR
               NOT ULTRAHIGH_ANN_GIT_STATUS STREQUAL "")
                set(ULTRAHIGH_ANN_GIT_DIRTY 1)
            endif()
        endif()
    endif()

    string(TOUPPER "${CMAKE_BUILD_TYPE}" ULTRAHIGH_ANN_BUILD_TYPE_UPPER)
    set(
        ULTRAHIGH_ANN_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${ULTRAHIGH_ANN_BUILD_TYPE_UPPER}}"
    )
    string(STRIP "${ULTRAHIGH_ANN_CXX_FLAGS}" ULTRAHIGH_ANN_CXX_FLAGS)
    set(
        ULTRAHIGH_ANN_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_${ULTRAHIGH_ANN_BUILD_TYPE_UPPER}}"
    )
    string(STRIP "${ULTRAHIGH_ANN_CUDA_FLAGS}" ULTRAHIGH_ANN_CUDA_FLAGS)
    if(ULTRAHIGH_ANN_ENABLE_CUDA)
        set(ULTRAHIGH_ANN_CUDA_ENABLED 1)
    else()
        set(ULTRAHIGH_ANN_CUDA_ENABLED 0)
    endif()

    foreach(variable IN ITEMS
            PROJECT_VERSION
            CMAKE_VERSION
            CMAKE_BUILD_TYPE
            CMAKE_CXX_COMPILER_ID
            CMAKE_CXX_COMPILER_VERSION
            CMAKE_CUDA_COMPILER_ID
            CMAKE_CUDA_COMPILER_VERSION
            CMAKE_CUDA_ARCHITECTURES
            CMAKE_SYSTEM_NAME
            CMAKE_SYSTEM_VERSION
            CMAKE_SYSTEM_PROCESSOR
            ULTRAHIGH_ANN_GIT_COMMIT
            ULTRAHIGH_ANN_CXX_FLAGS
            ULTRAHIGH_ANN_CUDA_FLAGS)
        string(REPLACE "\\" "\\\\" ${variable} "${${variable}}")
        string(REPLACE "\"" "\\\"" ${variable} "${${variable}}")
        string(REPLACE "\n" "\\n" ${variable} "${${variable}}")
    endforeach()

    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ultrahigh_ann_build_config.hpp.in"
        "${CMAKE_CURRENT_BINARY_DIR}/generated/ultrahigh_ann_build_config.hpp"
        @ONLY
    )

    add_executable(
        nearest_neighbor_benchmark
        src/benchmarks/benchmark_dataset.cpp
        src/benchmarks/benchmark_metrics.cpp
        src/benchmarks/benchmark_setup.cpp
        src/benchmarks/nearest_neighbor_benchmark.cpp
    )

    target_link_libraries(
        nearest_neighbor_benchmark
        PRIVATE
            ultrahigh_ann
    )

    target_include_directories(
        nearest_neighbor_benchmark
        PRIVATE
            "${CMAKE_CURRENT_BINARY_DIR}/generated"
    )

endif()

if(ULTRAHIGH_ANN_BUILD_TOOLS)
    add_executable(
        compute_l1_probabilities
        src/tools/compute_importance_probabilities.cpp
        src/tools/compute_l1_probabilities.cpp
    )

    target_link_libraries(
        compute_l1_probabilities
        PRIVATE
            ultrahigh_ann
            OpenMP::OpenMP_CXX
    )

    add_executable(
        compute_l2_probabilities
        src/tools/compute_importance_probabilities.cpp
        src/tools/compute_l2_probabilities.cpp
    )

    target_link_libraries(
        compute_l2_probabilities
        PRIVATE
            ultrahigh_ann
            OpenMP::OpenMP_CXX
    )
endif()

target_compile_features(ultrahigh_ann PUBLIC cxx_std_20)
target_include_directories(
    ultrahigh_ann
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

include(CMakePackageConfigHelpers)

set(
    ULTRAHIGH_ANN_INSTALL_CMAKEDIR
    "${CMAKE_INSTALL_LIBDIR}/cmake/ultrahigh_ann"
    CACHE STRING
    "Directory in which to install the ultrahigh_ann CMake package"
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ultrahigh_annConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ultrahigh_annConfig.cmake"
    INSTALL_DESTINATION "${ULTRAHIGH_ANN_INSTALL_CMAKEDIR}"
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ultrahigh_annConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMinorVersion
)

install(
    TARGETS ultrahigh_ann
    EXPORT ultrahigh_annTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
)

install(
    EXPORT ultrahigh_annTargets
    FILE ultrahigh_annTargets.cmake
    NAMESPACE ultrahigh_ann::
    DESTINATION "${ULTRAHIGH_ANN_INSTALL_CMAKEDIR}"
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/ultrahigh_annConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/ultrahigh_annConfigVersion.cmake"
    DESTINATION "${ULTRAHIGH_ANN_INSTALL_CMAKEDIR}"
)

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    include(CTest)
    set(ULTRAHIGH_ANN_BUILD_TESTS_DEFAULT "${BUILD_TESTING}")
else()
    set(ULTRAHIGH_ANN_BUILD_TESTS_DEFAULT OFF)
endif()

option(
    ULTRAHIGH_ANN_BUILD_TESTS
    "Build this project's CTest targets"
    "${ULTRAHIGH_ANN_BUILD_TESTS_DEFAULT}"
)
unset(ULTRAHIGH_ANN_BUILD_TESTS_DEFAULT)

option(
    ULTRAHIGH_ANN_REQUIRE_CUDA_DEVICE_TESTS
    "Treat an unavailable CUDA device as a test failure instead of a skip"
    OFF
)

if(ULTRAHIGH_ANN_BUILD_TESTS)
    if(ULTRAHIGH_ANN_REQUIRE_CUDA_DEVICE_TESTS AND
       NOT ULTRAHIGH_ANN_ENABLE_CUDA)
        message(
            FATAL_ERROR
            "ULTRAHIGH_ANN_REQUIRE_CUDA_DEVICE_TESTS=ON requires "
            "ULTRAHIGH_ANN_ENABLE_CUDA=ON"
        )
    endif()
    enable_testing()
    add_executable(
        benchmark_dataset_test
        src/benchmarks/benchmark_dataset.cpp
        tests/benchmarks/benchmark_dataset_test.cpp
    )

    target_link_libraries(
        benchmark_dataset_test
        PRIVATE
            ultrahigh_ann
    )

    target_include_directories(
        benchmark_dataset_test
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src/benchmarks"
    )

    target_compile_features(benchmark_dataset_test PRIVATE cxx_std_20)

    add_test(
        NAME benchmark_dataset
        COMMAND benchmark_dataset_test
    )

    add_executable(
        benchmark_metrics_test
        src/benchmarks/benchmark_metrics.cpp
        tests/benchmarks/benchmark_metrics_test.cpp
    )

    target_link_libraries(
        benchmark_metrics_test
        PRIVATE
            ultrahigh_ann
    )

    target_include_directories(
        benchmark_metrics_test
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src/benchmarks"
    )

    target_compile_features(benchmark_metrics_test PRIVATE cxx_std_20)

    add_test(
        NAME benchmark_metrics
        COMMAND benchmark_metrics_test
    )

    add_executable(
        benchmark_setup_test
        src/benchmarks/benchmark_setup.cpp
        tests/benchmarks/benchmark_setup_test.cpp
    )

    target_include_directories(
        benchmark_setup_test
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src/benchmarks"
    )

    target_compile_features(benchmark_setup_test PRIVATE cxx_std_20)

    add_test(
        NAME benchmark_setup
        COMMAND benchmark_setup_test
    )

    add_executable(
        representative_query_dataset_test
        tests/datasets/representative_query_dataset_test.cpp
    )

    target_link_libraries(
        representative_query_dataset_test
        PRIVATE
            ultrahigh_ann
    )

    add_test(
        NAME representative_query_dataset
        COMMAND representative_query_dataset_test
    )

    add_executable(
        importance_probability_io_test
        tests/io/importance_probability_io_test.cpp
    )

    target_link_libraries(
        importance_probability_io_test
        PRIVATE
            ultrahigh_ann
    )

    add_test(
        NAME importance_probability_io
        COMMAND importance_probability_io_test
    )

    add_executable(
        dense_matrix_test
        tests/core/dense_matrix_test.cpp
    )

    target_link_libraries(
        dense_matrix_test
        PRIVATE
            ultrahigh_ann
    )

    target_include_directories(
        dense_matrix_test
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )

    add_test(
        NAME dense_matrix_multiply_right_transposed
        COMMAND dense_matrix_test
    )

    add_executable(
        exact_l1_index_test
        tests/exact/l1/exact_l1_index_test.cpp
    )

    target_link_libraries(
        exact_l1_index_test
        PRIVATE
            ultrahigh_ann
    )

    add_test(
        NAME exact_l1_index
        COMMAND exact_l1_index_test
    )

    add_executable(
        cuda_exact_l1_index_test
        tests/exact/l1/cuda_exact_l1_index_test.cpp
    )

    target_link_libraries(
        cuda_exact_l1_index_test
        PRIVATE
            ultrahigh_ann
    )

    add_test(
        NAME cuda_exact_l1_index
        COMMAND cuda_exact_l1_index_test
    )

    add_executable(
        exact_l2_index_test
        tests/exact/l2/exact_l2_index_test.cpp
    )

    target_link_libraries(
        exact_l2_index_test
        PRIVATE
            ultrahigh_ann
    )

    add_test(
        NAME exact_l2_index
        COMMAND exact_l2_index_test
    )

    add_executable(
        cuda_exact_l2_index_test
        tests/exact/l2/cuda_exact_l2_index_test.cpp
    )

    target_link_libraries(
        cuda_exact_l2_index_test
        PRIVATE
            ultrahigh_ann
    )

    add_test(
        NAME cuda_exact_l2_index
        COMMAND cuda_exact_l2_index_test
    )

    add_executable(
        importance_sampling_test
        tests/hnsvw25/l1/importance_sampling_test.cpp
    )

    target_link_libraries(
        importance_sampling_test
        PRIVATE
            ultrahigh_ann
    )

    target_include_directories(
        importance_sampling_test
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )

    add_test(
        NAME importance_sampling_probability_one
        COMMAND importance_sampling_test
    )

    add_executable(
        l2_importance_sampling_test
        tests/hnsvw25/l2/importance_sampling_test.cpp
    )

    target_link_libraries(
        l2_importance_sampling_test
        PRIVATE
            ultrahigh_ann
    )

    target_include_directories(
        l2_importance_sampling_test
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )

    add_test(
        NAME l2_importance_sampling_probabilities
        COMMAND l2_importance_sampling_test
    )

    set_tests_properties(
        importance_sampling_probability_one
        l2_importance_sampling_probabilities
        PROPERTIES
            ENVIRONMENT "OMP_NUM_THREADS=4;OMP_DYNAMIC=FALSE"
    )

    add_executable(
        uniform_probabilities_test
        tests/coordinate_sampling/uniform_probabilities_test.cpp
    )

    target_link_libraries(
        uniform_probabilities_test
        PRIVATE
            ultrahigh_ann
    )

    add_test(
        NAME uniform_sampling_mass
        COMMAND uniform_probabilities_test
    )

    add_executable(
        uniform_ann_index_test
        tests/coordinate_sampling/uniform_ann_index_test.cpp
    )

    target_link_libraries(
        uniform_ann_index_test
        PRIVATE
            ultrahigh_ann
    )

    add_test(
        NAME uniform_ann_indexes
        COMMAND uniform_ann_index_test
    )

    add_executable(
        sampled_coordinate_ann_index_test
        tests/coordinate_sampling/sampled_coordinate_ann_index_test.cpp
    )

    target_link_libraries(
        sampled_coordinate_ann_index_test
        PRIVATE
            ultrahigh_ann
    )

    add_test(
        NAME sampled_coordinate_ann_indexes
        COMMAND sampled_coordinate_ann_index_test
    )

    add_executable(
        sampled_coordinate_l2_embedding_test
        tests/coordinate_sampling/sampled_coordinate_l2_embedding_test.cpp
    )

    target_link_libraries(
        sampled_coordinate_l2_embedding_test
        PRIVATE
            ultrahigh_ann
    )

    add_test(
        NAME sampled_coordinate_l2_embedding
        COMMAND sampled_coordinate_l2_embedding_test
    )

    add_executable(
        cuda_sampled_coordinate_l2_ann_index_test
        tests/coordinate_sampling/cuda_sampled_coordinate_l2_ann_index_test.cpp
    )

    target_link_libraries(
        cuda_sampled_coordinate_l2_ann_index_test
        PRIVATE
            ultrahigh_ann
    )

    add_test(
        NAME cuda_sampled_coordinate_l2_ann_index
        COMMAND cuda_sampled_coordinate_l2_ann_index_test
    )

    add_executable(
        cuda_sampled_coordinate_l1_ann_index_test
        tests/coordinate_sampling/cuda_sampled_coordinate_l1_ann_index_test.cpp
    )

    target_link_libraries(
        cuda_sampled_coordinate_l1_ann_index_test
        PRIVATE
            ultrahigh_ann
    )

    add_test(
        NAME cuda_sampled_coordinate_l1_ann_index
        COMMAND cuda_sampled_coordinate_l1_ann_index_test
    )

    add_executable(
        flat_l1_ann_index_test
        tests/hnsvw25/l1/flat/l1_ann_index_test.cpp
    )

    target_link_libraries(
        flat_l1_ann_index_test
        PRIVATE
            ultrahigh_ann
    )

    add_test(
        NAME flat_l1_ann_index_known_queries
        COMMAND flat_l1_ann_index_test
    )

    add_executable(
        flat_l2_ann_index_test
        tests/hnsvw25/l2/flat/l2_ann_index_test.cpp
    )

    target_link_libraries(
        flat_l2_ann_index_test
        PRIVATE
            ultrahigh_ann
    )

    add_test(
        NAME flat_l2_ann_index_known_queries
        COMMAND flat_l2_ann_index_test
    )

    set_tests_properties(
        exact_l2_index
        uniform_ann_indexes
        sampled_coordinate_l2_embedding
        flat_l2_ann_index_known_queries
        PROPERTIES
            ENVIRONMENT "OMP_NUM_THREADS=4;OMP_DYNAMIC=FALSE"
    )

    add_executable(
        hierarchical_l1_ann_index_test
        tests/hnsvw25/l1/hierarchical/hierarchical_l1_ann_index_test.cpp
    )

    target_link_libraries(
        hierarchical_l1_ann_index_test
        PRIVATE
            ultrahigh_ann
    )

    add_test(
        NAME hierarchical_l1_ann_index_known_queries
        COMMAND hierarchical_l1_ann_index_test
    )

    add_executable(
        cuda_hierarchical_l1_ann_index_test
        tests/hnsvw25/l1/hierarchical/cuda_hierarchical_l1_ann_index_test.cpp
    )

    target_link_libraries(
        cuda_hierarchical_l1_ann_index_test
        PRIVATE
            ultrahigh_ann
    )

    add_test(
        NAME cuda_hierarchical_l1_ann_index
        COMMAND cuda_hierarchical_l1_ann_index_test
    )

    add_executable(
        hierarchical_l2_ann_index_test
        tests/hnsvw25/l2/hierarchical/hierarchical_l2_ann_index_test.cpp
    )

    target_link_libraries(
        hierarchical_l2_ann_index_test
        PRIVATE
            ultrahigh_ann
    )

    add_test(
        NAME hierarchical_l2_ann_index_known_queries
        COMMAND hierarchical_l2_ann_index_test
    )

    add_executable(
        cuda_hierarchical_l2_ann_index_test
        tests/hnsvw25/l2/hierarchical/cuda_hierarchical_l2_ann_index_test.cpp
    )

    target_link_libraries(
        cuda_hierarchical_l2_ann_index_test
        PRIVATE
            ultrahigh_ann
    )

    add_test(
        NAME cuda_hierarchical_l2_ann_index
        COMMAND cuda_hierarchical_l2_ann_index_test
    )

    set(
        ULTRAHIGH_ANN_CUDA_DEVICE_TEST_TARGETS
        cuda_exact_l1_index_test
        cuda_exact_l2_index_test
        importance_sampling_test
        l2_importance_sampling_test
        cuda_sampled_coordinate_l1_ann_index_test
        cuda_sampled_coordinate_l2_ann_index_test
        cuda_hierarchical_l1_ann_index_test
        cuda_hierarchical_l2_ann_index_test
    )
    if(ULTRAHIGH_ANN_ENABLE_CUDA)
        foreach(target IN LISTS ULTRAHIGH_ANN_CUDA_DEVICE_TEST_TARGETS)
            target_compile_definitions(
                "${target}"
                PRIVATE
                    ULTRAHIGH_ANN_TEST_REQUIRES_CUDA_DEVICE=1
            )
        endforeach()
    endif()
    unset(ULTRAHIGH_ANN_CUDA_DEVICE_TEST_TARGETS)

    set_tests_properties(
        cuda_exact_l1_index
        cuda_exact_l2_index
        importance_sampling_probability_one
        l2_importance_sampling_probabilities
        cuda_sampled_coordinate_l1_ann_index
        cuda_sampled_coordinate_l2_ann_index
        cuda_hierarchical_l1_ann_index
        cuda_hierarchical_l2_ann_index
        PROPERTIES
            LABELS "cuda-device"
    )
    if(NOT ULTRAHIGH_ANN_REQUIRE_CUDA_DEVICE_TESTS)
        set_tests_properties(
            cuda_exact_l1_index
            cuda_exact_l2_index
            importance_sampling_probability_one
            l2_importance_sampling_probabilities
            cuda_sampled_coordinate_l1_ann_index
            cuda_sampled_coordinate_l2_ann_index
            cuda_hierarchical_l1_ann_index
            cuda_hierarchical_l2_ann_index
            PROPERTIES
                SKIP_RETURN_CODE 77
        )
    endif()
endif()
