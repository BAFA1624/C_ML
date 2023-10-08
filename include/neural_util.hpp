#pragma once

#include "Eigen/Dense"

#include <complex>
#include <functional>
#include <utility>
#include <vector>

namespace neural
{

template <typename T>
struct is_complex : std::false_type
{};
template <std::floating_point T>
struct is_complex<std::complex<T>> : std::true_type
{};

template <typename T>
concept Weight = std::floating_point<T> || is_complex<T>::value;

template <Weight T>
using layer_t = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <Weight T>
using network_t = std::vector<layer_t<T>>;

template <Weight T>
using output_layer_t = layer_t<T>;
template <Weight T>
using output_network_t = std::vector<layer_t<T>>;

template <Weight T>
using function_t =
    std::function<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(
        layer_t<T> & )>;
template <Weight T>
using cost_function_t = std::function<layer_t<T>( layer_t<T> &, layer_t<T> & )>;

} // namespace neural
