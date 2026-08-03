#pragma once

#include <concepts>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace ultrahigh_ann {

template<std::floating_point T>
class BasicDenseMatrix {
public:
    BasicDenseMatrix() = default;

    BasicDenseMatrix(
        std::vector<T>&& raw_data,
        std::size_t rows,
        std::size_t cols)
        : rows_(rows),
          cols_(cols),
          data_(std::move(raw_data))
    {
        if (cols_ != 0 && rows_ > data_.max_size() / cols_) {
            throw std::length_error(
                "DenseMatrix dimensions overflow");
        }
        if (data_.size() != rows_ * cols_) {
            throw std::invalid_argument(
                "DenseMatrix dimensions do not match the data size");
        }
    }

    BasicDenseMatrix(const BasicDenseMatrix&) = delete;
    BasicDenseMatrix& operator=(const BasicDenseMatrix&) = delete;

    BasicDenseMatrix(BasicDenseMatrix&& other) noexcept
        : rows_(std::exchange(other.rows_, 0)),
          cols_(std::exchange(other.cols_, 0)),
          data_(std::move(other.data_))
    {
    }

    BasicDenseMatrix& operator=(BasicDenseMatrix&& other) noexcept
    {
        if (this != &other) {
            rows_ = std::exchange(other.rows_, 0);
            cols_ = std::exchange(other.cols_, 0);
            data_ = std::move(other.data_);
        }

        return *this;
    }

    ~BasicDenseMatrix() = default;

    template<std::floating_point Right>
    [[nodiscard]] static auto multiply_right_transposed(
        const BasicDenseMatrix& left,
        const BasicDenseMatrix<Right>& right)
        -> BasicDenseMatrix<std::common_type_t<T, Right>>
    {
        if (left.cols() != right.cols()) {
            throw std::invalid_argument(
                "matrix inner dimensions do not match");
        }

        using Result = std::common_type_t<T, Right>;
        using Accumulator =
            std::common_type_t<double, T, Right>;

        const std::size_t output_rows = left.rows();
        const std::size_t output_cols = right.rows();
        const std::size_t inner_dimension = left.cols();

        const std::size_t maximum_size =
            std::vector<Result>{}.max_size();
        if (output_cols != 0 &&
            output_rows > maximum_size / output_cols) {
            throw std::length_error(
                "matrix result dimensions overflow");
        }

        std::vector<Result> output(
            output_rows * output_cols,
            Result{});

        for (std::size_t row_index = 0;
             row_index < output_rows;
             ++row_index) {
            const auto left_row = left.row(row_index);

            for (std::size_t output_column = 0;
                 output_column < output_cols;
                 ++output_column) {
                const auto right_row =
                    right.row(output_column);
                Accumulator sum{};

                for (std::size_t inner_index = 0;
                     inner_index < inner_dimension;
                     ++inner_index) {
                    sum +=
                        static_cast<Accumulator>(
                            left_row[inner_index]) *
                        static_cast<Accumulator>(
                            right_row[inner_index]);
                }

                output[
                    row_index * output_cols +
                    output_column] =
                        static_cast<Result>(sum);
            }
        }

        return BasicDenseMatrix<Result>(
            std::move(output),
            output_rows,
            output_cols);
    }

    [[nodiscard]] std::size_t rows() const noexcept
    {
        return rows_;
    }

    [[nodiscard]] std::size_t cols() const noexcept
    {
        return cols_;
    }

    [[nodiscard]] std::span<const T> values() const noexcept
    {
        return data_;
    }

    [[nodiscard]] std::span<const T> row(
        std::size_t index) const
    {
        if (index >= rows_) {
            throw std::out_of_range(
                "DenseMatrix row index out of range");
        }

        return std::span<const T>{data_}.subspan(
            index * cols_,
            cols_);
    }

private:
    std::size_t rows_{};
    std::size_t cols_{};
    std::vector<T> data_;
};

using DenseMatrix = BasicDenseMatrix<float>;
using DoubleDenseMatrix = BasicDenseMatrix<double>;

}  // namespace ultrahigh_ann
