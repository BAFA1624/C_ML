#pragma once // NEURAL_NETWORK_HPP

#include <cmath>
#include <complex>
#include <concepts>
#include <cstdlib>
#include <ctime>
#include <functional>
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
T
sigmoid( const T x ) {
    static_cast<T>( 1 ) / ( 1 + std::exp( -x ) );
}

template <weight_type T>
using bias_t = T;
template <weight_type T>
using layer_bias_t = std::vector<bias_t<T>>;
template <weight_type T>
using network_bias_t = std::vector<layer_bias_t<T>>;

template <weight_type T>
using neuron_weight_t = std::vector<T>;
template <weight_type T>
using layer_weight_t = std::vector<neuron_weight_t<T>>;
template <weight_type T>
using network_weight_t = std::vector<layer_weight_t<T>>;

template <weight_type T>
using function_t = std::function<T( const T )>;

template <weight_type T>
class NeuralNetwork
{
    private:
    std::uint64_t              m_n_inputs;
    std::uint64_t              m_n_outputs;
    std::uint64_t              m_n_layers;
    std::vector<std::uint64_t> m_neurons_per_layer;
    std::vector<function_t<T>> m_activation_functions;

    network_bias_t<T>   m_network_biases;
    network_weight_t<T> m_network_weights;

    public:
    NeuralNetwork( const std::uint64_t n_inputs, const std::uint64_t n_outputs,
                   const std::vector<std::uint64_t> & neurons_per_layer,
                   const std::vector<function_t<T>> & activation_functions,
                   const std::time_t                  seed = 1 ) :
        m_n_inputs( n_inputs ),
        m_n_outputs( n_outputs ),
        m_n_layers( neurons_per_layer.size() ),
        m_activation_functions( activation_functions ),
        m_neurons_per_layer( neurons_per_layer ) {
        // Must be an activation function for each layer
        assert( m_activation_functions.size() == m_n_layers );

        // Seed random number generator
        std::srand( seed );

        // Initializing network biases
        m_network_biases = network_bias_t<T>( m_n_layers );
        for ( std::uint64_t i{ 0 }; i < m_n_layers; ++i ) {
            m_network_biases[i] = layer_bias_t<T>( m_neurons_per_layer[i] );
        }

        // Initializing network weights
        m_network_weights = network_weight_t<T>( m_n_layers );
        for ( std::uint64_t i{ 0 }; i < m_n_layers; ++i ) {
            m_network_weights[i] = layer_weight_t<T>( m_neurons_per_layer[i] );
            for ( std::uint64_t j{ 0 }; j < m_neurons_per_layer[i]; ++j ) {
                if ( i == 0 ) {
                    m_network_weights[0][j] = neuron_weight_t<T>( m_n_inputs );
                }
                else {
                    m_network_weights[i][j] =
                        neuron_weight_t<T>( m_neurons_per_layer[i - 1] );
                }

                // Randomly initialize each weight
                for ( auto & weight : m_network_weights[i][j] ) {
                    weight = static_cast<T>( std::rand() ) / RAND_MAX;
                }
            }
        }
    }
};

} // namespace neural