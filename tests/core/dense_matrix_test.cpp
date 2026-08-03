#include "core/dense_matrix.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace {

bool expect(bool condition, std::string_view message)
{
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
    }

    return condition;
}

}  // namespace

int main()
{
    using ultrahigh_ann::DenseMatrix;
    using ultrahigh_ann::DoubleDenseMatrix;

    std::vector<float> left_values{
        1.0F, 2.0F, 3.0F,
        4.0F, 5.0F, 6.0F};
    const DenseMatrix left(
        std::move(left_values),
        2,
        3);

    std::vector<double> right_values{
        7.0, 8.0, 9.0,
        -1.0, 0.0, 1.0};
    const DoubleDenseMatrix right(
        std::move(right_values),
        2,
        3);

    const DoubleDenseMatrix product =
        DenseMatrix::multiply_right_transposed(left, right);

    bool passed = true;
    passed &= expect(
        product.rows() == 2 && product.cols() == 2,
        "the result must have one row per left row and one column per right row");

    constexpr double tolerance = 1.0e-12;
    if (product.rows() == 2 && product.cols() == 2) {
        passed &= expect(
            std::abs(product.row(0)[0] - 50.0) < tolerance,
            "the first dot product must equal 50");
        passed &= expect(
            std::abs(product.row(0)[1] - 2.0) < tolerance,
            "the second dot product must equal 2");
        passed &= expect(
            std::abs(product.row(1)[0] - 122.0) < tolerance,
            "the third dot product must equal 122");
        passed &= expect(
            std::abs(product.row(1)[1] - 2.0) < tolerance,
            "the fourth dot product must equal 2");
    }

    std::vector<double> incompatible_values{
        1.0, 2.0};
    const DoubleDenseMatrix incompatible(
        std::move(incompatible_values),
        1,
        2);

    bool dimension_error_thrown = false;
    try {
        static_cast<void>(
            DenseMatrix::multiply_right_transposed(
                left,
                incompatible));
    } catch (const std::invalid_argument&) {
        dimension_error_thrown = true;
    }

    passed &= expect(
        dimension_error_thrown,
        "incompatible row lengths must throw invalid_argument");

    return passed ? 0 : 1;
}
