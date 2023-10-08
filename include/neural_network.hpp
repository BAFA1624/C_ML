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
    cost_function_t<T>         m_cost_function;

    network_t<T>        m_network;
    output_network_t<T> m_intermediate_state;

    public:
    NeuralNetwork( const std::vector<std::uint64_t> & neurons_per_layer,
                   const std::vector<function_t<T>> & activation_functions,
                   const std::vector<function_t<T>> & activation_gradients,
                   const cost_function_t<T> & cost_function = cost::SSR<T>,
                   const std::time_t          seed = 1 ) :
        m_n_inputs( neurons_per_layer.front() ),
        m_n_outputs( neurons_per_layer.back() ),
        m_n_layers( neurons_per_layer.size() ),
        m_neurons_per_layer( neurons_per_layer ),
        m_cost_function( cost_function ) {
        // Must be an activation function for each layer (except the input
        // layer)
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
        std::uint64_t prev_layer_sz{ m_n_inputs };
        for ( std::uint64_t i{ 0 }; i < m_n_layers; ++i ) {
            const auto n_neurons{ m_neurons_per_layer[i] };
            m_network[i] = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(
                prev_layer_sz, n_neurons );
            m_network[i].setRandom();
            prev_layer_sz = n_neurons;
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
    [[nodiscard]] constexpr inline auto cost_func() const noexcept {
        return m_cost_function;
    }
    [[nodiscard]] constexpr inline auto & network() const noexcept {
        return m_network;
    }
    [[nodiscard]] constexpr inline auto & intermediate_state() const noexcept {
        return m_intermediate_state;
    }
    [[nodiscard]] constexpr inline auto &
    layer( const std::uint64_t i ) const noexcept {
        return m_network.at( i );
    }

    [[nodiscard]] constexpr inline Eigen::Matrix<T, Eigen::Dynamic,
                                                 Eigen::Dynamic>
    forward_pass( const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &
                      inputs ) noexcept;
    [[nodiscard]] constexpr inline auto
    backward_pass( const std::vector<T> & labels ) noexcept;
};

template <Weight T>
[[nodiscard]] constexpr inline Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
NeuralNetwork<T>::forward_pass(
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> & inputs ) noexcept {
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> input( inputs.rows(),
                                                            inputs.cols() );
    input << inputs;
    assert( input.cols() == m_network.front().rows() );
    m_intermediate_state[0] = input;
    const auto view = std::views::zip( m_network, m_intermediate_state,
                                       m_activation_functions )
                      | std::views::drop( 1 );
    for ( const auto & [i, layers] : view | std::views::enumerate ) {
        auto & [network_layer, output_layer, act_func] = layers;

        input = input * network_layer;
        input = act_func( input );

        output_layer = input;
    }

    return input;
}

template <Weight T>
[[nodiscard]] constexpr inline auto
NeuralNetwork<T>::backward_pass( const std::vector<T> & labels ) noexcept {
    auto       expected_output{ labels };
    const auto layer_info{ std::views::zip( m_network, m_intermediate_state,
                                            m_activation_gradients )
                           | std::views::drop( 1 ) | std::views::reverse };
    const auto layer_info_view{ layer_info | std::views::slide( 2 ) };

    for ( const auto & [i, layers] : layer_info_view | std::views::enumerate ) {
        const auto & nxt_layer = layers[0];
        const auto & cur_layer = layers[1];
        const auto & [nxt_weights, nxt_output, nxt_gradient] = nxt_layer;
        const auto & [cur_weights, cur_output, cur_gradient] = cur_layer;

        const auto cost{ m_cost_function( cur_output, expected_output ) };
    }

    for ( const auto & [layer, output_layer, gradient_func] :
          layer_view | std::views::enumerate ) {
        for ( const auto & [j, layers] :
              std::views::zip( layer, output_layer ) | std::views::enumerate ) {
            const auto & [layer, output_layer] = layers;
            // Calculate error per weight & bias
            const auto cost = m_cost_function( output_layer, expected_output );
            // Calculate gradients
            const auto d_weights = 2. * gradient_func( layer );
            // Adjust weights & bias

            // Update expected output for next layer
            expected_output = ;
        }
    }
}


} // namespace neural
