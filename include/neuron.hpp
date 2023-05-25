#pragma once

#include "activation_functions.hpp"

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
    T                       m_weight;
    std::vector<Neuron<T> > m_outputs;
    activation_func<T>      m_activation;

    public:
    Neuron( const T initial_weight, const activation_func<T> f ) :
        m_weight( initial_weight ), m_activation( f ) {}

    virtual void add_outputs( const std::vector<Neuron<T> > & outputs ) {
        m_outputs.insert( m_outputs.end(), outputs.begin(), outputs.end() );
    };
};

template <std::floating_point T>
class OutputNeuron : public Neuron<T>
{
    public:
    void
    add_outputs( const std::vector<Neuron<T> > & outputs ) override = delete;
};
