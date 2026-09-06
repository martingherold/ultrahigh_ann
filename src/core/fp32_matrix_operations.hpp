#pragma once

#include "core/finite_values.hpp"
#include "ultrahigh_ann/core/dense_matrix.hpp"

#include <cstddef>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace ultrahigh_ann::detail {

[[nodiscard]] inline DenseMatrix narrow_matrix_to_fp32(
    const DoubleDenseMatrix& source, std::string_view value_name)
{
    std::vector<float> values;
    values.reserve(source.values().size());
    for (const double source_value : source.values()) {
        const float value = static_cast<float>(source_value);
        validate_finite_result(value, value_name);
        values.push_back(value);
    }
    return DenseMatrix(std::move(values), source.rows(), source.cols());
}

// This is deliberately separate from BasicDenseMatrix's binary64-accumulating
// product. Both the products and accumulator remain binary32.
[[nodiscard]] inline DenseMatrix multiply_right_transposed_fp32(
    const DenseMatrix& left, const DenseMatrix& right,
    std::string_view result_name)
{
    if (left.cols() != right.cols()) {
        throw std::invalid_argument(
            "matrix inner dimensions do not match");
    }

    const std::size_t output_rows = left.rows();
    const std::size_t output_cols = right.rows();
    const std::size_t inner_dimension = left.cols();
    const std::size_t maximum_size = std::vector<float>{}.max_size();
    if (output_cols != 0 && output_rows > maximum_size / output_cols) {
        throw std::length_error("matrix result dimensions overflow");
    }

    std::vector<float> output(output_rows * output_cols, 0.0F);
    for (std::size_t row_index = 0;
         row_index < output_rows;
         ++row_index) {
        const auto left_row = left.row(row_index);
        for (std::size_t output_column = 0;
             output_column < output_cols;
             ++output_column) {
            const auto right_row = right.row(output_column);
            float sum = 0.0F;
            for (std::size_t inner_index = 0;
                 inner_index < inner_dimension;
                 ++inner_index) {
                sum += left_row[inner_index] * right_row[inner_index];
            }
            validate_finite_result(sum, result_name);
            output[row_index * output_cols + output_column] = sum;
        }
    }

    return DenseMatrix(std::move(output), output_rows, output_cols);
}

}  // namespace ultrahigh_ann::detail
