#include "coordinate_sampling/filter_columns.hpp"
#include "ultrahigh_ann/core/dense_matrix.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

namespace ultrahigh_ann
{
[[nodiscard]] DenseMatrix filter_columns(
const DenseMatrix& representatives,
std::span<const SampledColumn> columns)
{
    const std::size_t rows{representatives.rows()};
    const std::size_t output_cols {columns.size()};
    const std::size_t original_cols{representatives.cols()};

    // Validate once, outside the hot loop.
    for (const SampledColumn& column : columns) {
        if (column.source_column >= original_cols) {
            throw std::out_of_range("sampled column index out of range");
        }
    }

    std::vector<float> output_data(rows*output_cols);
    std::size_t output_index {0};

    for(std::size_t row_index = 0 ; row_index < rows ; ++row_index){
        const auto row = representatives.row(row_index);
        for(const SampledColumn& column: columns){
            output_data[output_index++] = row[column.source_column];
        }
    }

    return DenseMatrix(std::move(output_data),rows,output_cols);
}



} // namespace ultrahigh_ann
