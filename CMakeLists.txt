cmake_minimum_required(VERSION 3.20)

project(
    ultrahigh_ann
    VERSION 0.1.0
    DESCRIPTION "Sublinear coordinate acquisition for nearest-representative search"
    LANGUAGES CXX
)

add_library(
    ultrahigh_ann
    src/datasets/representative_query_dataset.cpp
    src/exact/l1/exact_l1_index.cpp
    src/exact/l2/exact_l2_index.cpp
    src/coordinate_sampling/coordinate_sampling.cpp
    src/coordinate_sampling/filter_columns.cpp
    src/coordinate_sampling/sampled_coordinate_l1_ann_index.cpp
    src/coordinate_sampling/sampled_coordinate_l2_ann_index.cpp
    src/coordinate_sampling/uniform_l1_ann_index.cpp
    src/coordinate_sampling/uniform_l2_ann_index.cpp
    src/coordinate_sampling/uniform_probabilities.cpp
    src/hnsvw25/l1/flat/l1_ann_index.cpp
    src/hnsvw25/l1/hierarchical/hierarchical_l1_ann_index.cpp
    src/hnsvw25/l1/importance_sampling.cpp
    src/hnsvw25/l2/flat/l2_ann_index.cpp
    src/hnsvw25/l2/hierarchical/hierarchical_l2_ann_index.cpp
    src/hnsvw25/l2/importance_sampling.cpp
    src/core/l1_distance.cpp
    src/core/l2_distance.cpp
)
add_library(ultrahigh_ann::ultrahigh_ann ALIAS ultrahigh_ann)

add_executable(
    nearest_representative_benchmark
    src/benchmarks/benchmark_metrics.cpp
    src/benchmarks/benchmark_setup.cpp
    src/benchmarks/nearest_representative_benchmark.cpp
)

target_link_libraries(
    nearest_representative_benchmark
    PRIVATE
        ultrahigh_ann
)

target_compile_features(ultrahigh_ann PUBLIC cxx_std_23)
target_include_directories(
    ultrahigh_ann
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

include(CTest)

if(BUILD_TESTING)
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

    target_compile_features(benchmark_metrics_test PRIVATE cxx_std_23)

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

    target_compile_features(benchmark_setup_test PRIVATE cxx_std_23)

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
        dense_matrix_test
        tests/core/dense_matrix_test.cpp
    )

    target_link_libraries(
        dense_matrix_test
        PRIVATE
            ultrahigh_ann
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
endif()
