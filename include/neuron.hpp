#pragma once

#include "activation_functions.hpp"

#include <cmath>
#include <complex>
#include <concepts>
#include <functional>
#include <memory>
#include <vector>


namespace C_ML
{

template <typename T>
struct is_complex : std::false_type
{};
template <std::floating_point T>
struct is_complex<std::complex<T>> : std::true_type
{};

namespace neuron
{

template <typename T>
concept neuron_weight = std::floating_point<T> || is_complex<T>::value;

template <neuron_weight T>
using activation_func = std::function<T( const T )>;

template <neuron_weight T>
class Neuron
{
    private:
    T                      m_weight;
    std::vector<Neuron<T>> m_outputs;
    activation_func<T>     m_activation;

    public:
    Neuron( const T initial_weight, const activation_func<T> f ) :
        m_weight( initial_weight ), m_activation( f ) {}
    virtual ~Neuron() = default;

    virtual void add_outputs( const std::vector<Neuron<T>> & outputs ) {
        m_outputs.insert( m_outputs.end(), outputs.begin(), outputs.end() );
    };
};

template <neuron_weight T>
class OutputNeuron final : public Neuron<T>
{
    public:
    void
    add_outputs( const std::vector<Neuron<T>> & outputs ) override = delete;
};

}; // namespace neuron

}; // namespace C_ML