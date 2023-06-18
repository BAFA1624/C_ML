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

template <weight_type T>
class NeuralNetwork
{
    private:
    using bias_t = T;
    using layer_bias_t = std::vector<bias_t>;
    using network_bias_t = std::vector<layer_bias_t>;

    using neuron_weight_t = std::vector<T>;
    using layer_weight_t = std::vector<neuron_weight_t>;
    using network_weight_t = std::vector<layer_weight_t>;

    std::uint64_t              m_n_inputs;
    std::uint64_t              m_n_outputs;
    std::uint64_t              m_n_layers;
    std::vector<std::uint64_t> m_neurons_per_layer;

    network_bias_t   m_network_biases;
    network_weight_t m_network_weights;
};

} // namespace neural