#pragma once

#include <array>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

namespace C_ML
{

template <typename T>
struct is_complex : std::false_type
{};
template <std::floating_point T>
struct is_complex<std::complex<T>> : std::true_type
{};

template <typename T>
concept param_type = std::floating_point<T> || is_complex<T>::value;

namespace neuron
{

template <typename T>
concept neuron_weight = param_type<T>;

template <neuron_weight T>
using activation_func = std::function<T( const T )>;

enum class initialisation_type { random };
using init_t = initialisation_type;

}; // namespace neuron

namespace layer
{

template <typename T>
concept neuron_weight = param_type<T>;

};

}; // namespace C_ML