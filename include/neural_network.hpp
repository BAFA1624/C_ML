#pragma once // NEURAL_NETWORK_HPP

#include <complex>
#include <concepts>
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
concept weight_type = std::floating_point<T> || is_complex<T>::value;

template <typename T>
class NeuralNetwork
{
    private:
    using bias_t = typename T;
    using layer_bias_t = typename std::vector<bias_t>;
    using network_bias_t = typename std::vector<layer_bias_t>;

    std::uint64_t m_n_inputs;
    std::uint64_t m_n_outputs;
    std::uint64_t m_n_layers;

    network_bias_t m_network_bias;
};

} // namespace neural