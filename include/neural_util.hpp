#pragma once

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

// Pair of weight & bias
template <Weight T>
using neuron_weight_t = std::vector<T>;

template <Weight T>
using neuron_t = std::pair<neuron_weight_t<T>, T>;

template <Weight T>
using layer_t = std::vector<neuron_t<T>>;
template <Weight T>
using network_t = std::vector<layer_t<T>>;

template <Weight T>
using output_layer_t = std::vector<T>;
template <Weight T>
using output_network_t = std::vector<output_layer_t<T>>;

template <Weight T>
using function_t = std::function<std::vector<T>( const std::vector<T> & )>;

} // namespace neural
