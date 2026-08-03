#pragma once

#include <cmath>
#include <concepts>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ultrahigh_ann::detail {

template<std::floating_point T>
void validate_finite_value(
    T value,
    std::string_view value_name)
{
    if (!std::isfinite(value)) {
        throw std::invalid_argument(
            std::string(value_name) +
            " must be finite");
    }
}

template<std::floating_point T>
void validate_finite_values(
    std::span<const T> values,
    std::string_view value_name)
{
    for (const T value : values) {
        validate_finite_value(value, value_name);
    }
}

template<std::floating_point T>
void validate_finite_result(
    T value,
    std::string_view value_name)
{
    if (!std::isfinite(value)) {
        throw std::overflow_error(
            std::string(value_name) +
            " is not finite");
    }
}

template<std::floating_point T>
void validate_finite_result(
    std::span<const T> values,
    std::string_view value_name)
{
    for (const T value : values) {
        validate_finite_result(value, value_name);
    }
}

template<std::floating_point T>
void validate_finite_domain_value(
    T value,
    std::string_view value_name)
{
    if (!std::isfinite(value)) {
        throw std::domain_error(
            std::string(value_name) +
            " must be finite");
    }
}

}  // namespace ultrahigh_ann::detail
