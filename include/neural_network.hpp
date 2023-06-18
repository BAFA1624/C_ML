#pragma once // NEURAL_NETWORK_HPP

#include <complex>
#include <concepts>

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
    std::uint64_t m_n_inputs;
    std::uint64_t m_n_outputs;
    std::uint64_t m_n_layers;
};

} // namespace neural