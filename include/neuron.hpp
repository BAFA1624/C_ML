#pragma once

#include <cmath>
#include <concepts>
#include <functional>
#include <memory>
#include <vector>

template <std::floating_point T>
using activation_func = std::function<T( const T )>;

template <std::floating_point T>
class Neuron
{
    private:
    T                        m_weight;
    std::vector<Neuron<T> &> m_outputs;
    activation_func<T>       m_activation;

    public:
};

template <std::floating_point T>
using OutputNeuron = Neuron<T>;