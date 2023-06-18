#pragma once // NEURAL_NETWORK_HPP

namespace neural
{

template <typename T>
struct is_complex : std::false_type
{};
template <std::floating_point T>
struct is_complex<std::complex<T>> : std::true_type
{};



template <typename T>
class NeuralNetwork
{};

} // namespace neural