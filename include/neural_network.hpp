#pragma once // NEURAL_NETWORK_HPP

#include "Eigen/Core"
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
#include <format>
#include <functional>
#include <iostream>
#include <numeric>
#include <ranges>
#include <vector>

namespace neural
{

template <Weight T>
class NeuralNetwork
{
    private:
    std::uint64_t              m_n_inputs;
    std::uint64_t              m_n_outputs;
    std::uint64_t              m_n_layers;
    std::vector<std::uint64_t> m_neurons_per_layer;
    std::vector<function_t<T>> m_activation_functions;
    std::vector<function_t<T>> m_activation_gradients;
    function_t<T>              m_cost_function;

    network_t<T>        m_network;
    output_network_t<T> m_intermediate_state;

    public:
    NeuralNetwork( const std::vector<std::uint64_t> & neurons_per_layer,
                   const std::vector<function_t<T>> & activation_functions,
                   const std::vector<function_t<T>> & activation_gradients,
                   const std::time_t                  seed = 1 ) :
        m_n_inputs( neurons_per_layer.front() ),
        m_n_outputs( neurons_per_layer.back() ),
        m_n_layers( neurons_per_layer.size() ),
        m_neurons_per_layer( neurons_per_layer ) {
        // Must be an activation function for each layer
        // (except the input layer)
        m_activation_functions = std::vector<function_t<T>>{};
        m_activation_functions.push_back( activation::linear<T> );
        m_activation_functions.insert( m_activation_functions.end(),
                                       activation_functions.cbegin(),
                                       activation_functions.cend() );
        m_activation_gradients = std::vector<function_t<T>>{};
        m_activation_gradients.push_back( activation::linear<T> );
        m_activation_gradients.insert( m_activation_gradients.end(),
                                       activation_gradients.cbegin(),
                                       activation_gradients.cend() );
        assert( m_activation_functions.size() == m_n_layers );

        // Seed random number generator
        std::srand( static_cast<unsigned>( seed ) );

        // Initializing network & intermediate_state in the same
        // shape for back propagation using the outputs of each layer
        m_network = network_t<T>( m_n_layers );
        m_intermediate_state = output_network_t<T>( m_n_layers );

        // Initializing layers
        for ( std::uint64_t i{ 0 }; i < m_n_layers; ++i ) {
            m_network[i] = layer_t<T>( m_neurons_per_layer[i] );
            m_intermediate_state[i] =
                output_layer_t<T>( m_neurons_per_layer[i] );
            for ( std::uint64_t j{ 0 }; j < m_neurons_per_layer[i]; ++j ) {
                if ( i == 0 ) {
                    m_network[0][j] =
                        neuron_t<T>{ neuron_weight_t<T>{ static_cast<T>( 1 ) },
                                     0 };
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
    [[nodiscard]] constexpr inline auto intermediate_state() const noexcept {
        return m_intermediate_state;
    }

    [[nodiscard]] constexpr inline auto
    forward_pass( const std::vector<T> & inputs ) noexcept;
    [[nodiscard]] constexpr inline auto
    backward_pass( const std::vector<T> & labels ) noexcept;
};

template <Weight T>
[[nodiscard]] constexpr inline auto
NeuralNetwork<T>::forward_pass( const std::vector<T> & inputs ) noexcept {
    assert( inputs.size() == m_n_inputs );
    m_intermediate_state[0] = inputs;
    const auto layer_view{ std::views::zip( m_network, m_intermediate_state,
                                            m_activation_functions )
                           | std::views::drop( 1 ) | std::views::enumerate };
    for ( const auto & [i, layer_data] : layer_view ) {
        const auto & [layer, output_layer, activation_func] = layer_data;
        const auto & prev_layer{ m_intermediate_state[i] };

        for ( const auto & [pair, output] :
              std::views::zip( layer, output_layer ) ) {
            const auto & [weights, bias] = pair;
            output =
                std::inner_product( weights.cbegin(), weights.cend(),
                                    prev_layer.cbegin(), static_cast<T>( 0 ) )
                + bias;
        }

        output_layer = activation_func( output_layer );
    }
}

template <Weight T>
[[nodiscard]] constexpr inline auto
NeuralNetwork<T>::backward_pass( const std::vector<T> & labels ) noexcept {
    auto       expected_output{ labels };
    const auto layer_view{ std::views::zip( m_network, m_intermediate_state,
                                            m_activation_gradients )
                           | std::views::drop( 1 ) | std::views::reverse };

    for ( const auto & [layer, output_layer, gradient_func] : layer_view ) {
        for ( const auto & [neuron, output] :
              std::views::zip( layer, output_layer ) ) {
            const auto & [weights, bias] = neuron;
            // Calculate error per weight & bias
            // Calculate gradients
            // Adjust weights & bias
            // Update expected output for next layer
        }
    }
}


} // namespace neural
