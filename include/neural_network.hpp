#pragma once // NEURAL_NETWORK_HPP

#include "neural_activation.hpp"
#include "neural_util.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <iostream>
#include <numeric>
#include <ranges>
#include <vector>

namespace neural
{

template <weight_type T>
class NeuralNetwork
{
    private:
    std::uint64_t              m_n_inputs;
    std::uint64_t              m_n_outputs;
    std::uint64_t              m_n_layers;
    std::vector<std::uint64_t> m_neurons_per_layer;
    std::vector<function_t<T>> m_activation_functions;

    network_t<T> m_network;

    public:
    NeuralNetwork( const std::vector<std::uint64_t> & neurons_per_layer,
                   const std::vector<function_t<T>> & activation_functions,
                   const std::time_t                  seed = 1 ) :
        m_n_inputs( neurons_per_layer.front() ),
        m_n_outputs( neurons_per_layer.back() ),
        m_n_layers( neurons_per_layer.size() ),
        m_neurons_per_layer( neurons_per_layer ) {
        // Must be an activation function for each layer
        // (except the input layer)
        m_activation_functions = std::vector<function_t<T>>{};
        m_activation_functions.push_back( linear<T> );
        m_activation_functions.insert( m_activation_functions.end(),
                                       activation_functions.cbegin(),
                                       activation_functions.cend() );
        assert( m_activation_functions.size() == m_n_layers );

        // Seed random number generator
        std::srand( static_cast<unsigned>( seed ) );

        // Initializing network
        m_network = network_t<T>( m_n_layers );

        // Initializing layers
        for ( std::uint64_t i{ 0 }; i < m_n_layers; ++i ) {
            m_network[i] = layer_t<T>( m_neurons_per_layer[i] );
            for ( std::uint64_t j{ 0 }; j < m_neurons_per_layer[i]; ++j ) {
                if ( i == 0 ) {
                    m_network[0][j] =
                        neuron_t<T>{ neuron_weight_t<T>{ static_cast<T>( 1 ) },
                                     0 };
                }
                else if ( i == 1 ) {
                    m_network[1][j] =
                        neuron_t<T>{ neuron_weight_t<T>( m_n_inputs ), 0 };
                }
                else {
                    m_network[i][j] = neuron_t<T>{
                        neuron_weight_t<T>( m_neurons_per_layer[i - 1] ), 0
                    };
                }
                // Randomly initialize each weight
                auto & [network_weights, neuron_bias] = m_network[i][j];
                std::generate(
                    network_weights.begin(), network_weights.end(),
                    []() { return static_cast<T>( std::rand() ) / RAND_MAX; } );
                neuron_bias = static_cast<T>( std::rand() ) / RAND_MAX;
            }
        }
    }

    [[nodiscard]] constexpr inline auto n_inputs() const noexcept {
        return m_n_inputs;
    }
    [[nodiscard]] constexpr inline auto n_outputs() const noexcept {
        return m_n_outputs;
    }
    [[nodiscard]] constexpr inline auto n_layers() const noexcept {
        return m_n_layers;
    }
    [[nodiscard]] constexpr inline auto shape() const noexcept {
        return m_neurons_per_layer;
    }
    [[nodiscard]] constexpr inline auto activation_funcs() const noexcept {
        return m_activation_functions;
    }
    [[nodiscard]] constexpr inline auto network() const noexcept {
        return m_network;
    }

    [[nodiscard]] constexpr inline auto
    forward( const std::vector<T> & inputs ) const noexcept;
};

template <weight_type T>
[[nodiscard]] constexpr inline auto
NeuralNetwork<T>::forward( const std::vector<T> & inputs ) const noexcept {
    std::vector<T> prev_layer_res{ inputs };
    for ( const auto & [layer, activation_func] :
          std::views::zip( m_network, m_activation_functions )
              | std::views::drop( 1 ) ) {
        std::vector<T> tmp( layer.size() );
        for ( const auto & [pair, x] : std::views::zip( layer, tmp ) ) {
            const auto & [weights, bias] = pair;
            x = std::inner_product( weights.cbegin(), weights.cend(),
                                    prev_layer_res.cbegin(),
                                    static_cast<T>( 0 ) )
                + bias;
        }
        prev_layer_res.resize( tmp.size() );
        std::ranges::transform( tmp.cbegin(), tmp.cend(),
                                prev_layer_res.begin(), activation_func );
    }
    return prev_layer_res;
}

} // namespace neural
