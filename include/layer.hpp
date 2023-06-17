#pragma once

#include "neuron.hpp"

namespace C_ML
{

namespace layer
{

template <neuron_weight T, activation_func<T> f,
          neuron::init_t Init = neuron::init_t::random>
class Layer
{
    private:
    std::vector<neuron::Neuron<T, f, Init> &> m_neurons;

    public:
    Layer( const neuron::activation_func<T>                       f,
           const std::array<neuron::Neuron<T, Init> &, InputSize> inputs ) :
        m_neurons( N ) {
        for ( auto & neuron : m_neurons ) {
            neuron = Neuron<T, Init>( f, inputs );
        }
    }

    virtual void feed_forward() {}
};

}; // namespace layer

}; // namespace C_ML