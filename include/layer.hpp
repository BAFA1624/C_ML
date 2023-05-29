#pragma once

#include "neuron.hpp"

namespace C_ML
{

namespace layer
{

template <neuron_weight T, neuron::init_t Init = neuron::init_t::random>
class Layer
{
    private:
    std::vector<neuron::Neuron<T, Init> &> m_neurons;

    public:
    Layer( const std::size_t N, const neuron::activation_func<T> f,
           const std::vector<neuron::Neuron<T, Init> &> inputs ) :
        m_neurons( N ) {
        for ( auto & neuron : m_neurons ) {
            neuron = Neuron<T, Init>( f, inputs );
        }
    }

    void feed_forward() {}
};

}; // namespace layer

}; // namespace C_ML