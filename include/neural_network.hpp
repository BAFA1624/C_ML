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
    cost_function_t<T>         m_cost_gradient;
    T                          m_eta;

    network_t<T>        m_network;
    output_network_t<T> m_intermediate_state;

    public:
    NeuralNetwork( const std::vector<std::uint64_t> & neurons_per_layer,
                   const std::vector<function_t<T>> & activation_functions,
                   const std::vector<function_t<T>> & activation_gradients,
                   const cost_function_t<T> & cost_function = cost::SSR<T>,
                   const cost_function_t<T> & cost_gradient = cost::d_SSR<T>,
                   const T                    eta = static_cast<T>( 0.01 ),
                   const std::time_t          seed = 1 ) :
        m_n_inputs( neurons_per_layer.front() ),
        m_n_outputs( neurons_per_layer.back() ),
        m_n_layers( neurons_per_layer.size() ),
        m_neurons_per_layer( neurons_per_layer ),
        m_cost_function( cost_function ),
        m_cost_gradient( cost_gradient ),
        m_eta( eta ) {
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

        // Initializing network, bisaes & intermediate_state in the same
        // shape for back propagation using the outputs of each layer
        m_network = network_t<T>( m_n_layers );
        m_intermediate_state = output_network_t<T>( m_n_layers );

        /*
        auto layer_info{ std::views::zip( m_network, m_intermediate_state,
                                   m_activation_gradients )
                  | std::views::reverse };
         auto layer_info_view{ layer_info | std::views::slide( 2 ) };
         */

        // Initializing layers
        /*const auto layer_info{ std::views::zip( m_neurons_per_layer,
        m_network, m_intermediate_state ) }; for ( const auto & layer_data :
        layer_info | std::views::slide( 2 ) ) { const auto & prev_layers =
        layer_data[0]; const auto & cur_layers = layer_data[1];

            const auto & [prev_n_neurons, prev_layer, prev_outputs] =
                prev_layers;
            const auto & [cur_n_neurons, cur_layer, cur_outputs] = cur_layers;

            cur_layer = layer_t<T>::Random( prev_n_neurons + 1, cur_n_neurons );
        }*/
        std::uint64_t prev_layer_sz{ m_n_inputs };

        for ( std::uint64_t i{ 0 }; i < m_n_layers; ++i ) {
            const auto n_neurons{ m_neurons_per_layer[i] };
            m_network[i] = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(
                prev_layer_sz + 1, n_neurons );
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
    [[nodiscard]] constexpr inline auto cost_gradient() const noexcept {
        return m_cost_gradient;
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
    forward_pass(
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> & inputs,
        const std::size_t V = 0 ) noexcept;
    [[nodiscard]] constexpr inline auto
    backward_pass( const layer_t<T> & labels,
                   const std::size_t  V = 0 ) noexcept;
};

template <Weight T>
[[nodiscard]] constexpr inline Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
NeuralNetwork<T>::forward_pass(
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> & inputs,
    const std::size_t                                        V ) noexcept {
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> input( inputs.rows(),
                                                            inputs.cols() + 1 );
    assert( input.cols() == m_network.front().rows() );
    const auto bias{ layer_t<T>::Constant( input.rows(), 1, 1. ) };
    input << inputs, bias;
    m_intermediate_state[0] = input;
    const auto view = std::views::zip( m_network, m_intermediate_state,
                                       m_activation_functions )
                      | std::views::drop( 1 );
    if ( V )
        std::cout << "Forward pass:\n";
    for ( const auto & [i, layers] : view | std::views::enumerate ) {
        auto & [network_layer, output_layer, act_func] = layers;

        output_layer = layer_t<T>::Constant(
            input.rows(), network_layer.cols() + 1, static_cast<T>( 1. ) );

        output_layer.leftCols( output_layer.cols() - 1 )
            << act_func( input * network_layer );

        // TODO: Remove when working.
        if ( V ) {
            std::cout << i << "\n";
            std::cout << std::format( "input ({}, {})\n", input.rows(),
                                      input.cols() )
                      << input << "\n";
            if ( V > 1 ) {
                std::cout << std::format( "network_layer ({}, {})\n",
                                          network_layer.rows(),
                                          network_layer.cols() )
                          << network_layer << "\n";
            }
            std::cout << std::format( "output_layer ({}, {})\n",
                                      output_layer.rows(), output_layer.cols() )
                      << output_layer << "\n";
        }
        input = output_layer;
    }
    if ( V ) {
        std::cout << std::format( "output ({}, {})\n", input.rows(),
                                  input.cols() )
                  << input << "\n";
        std::cout << "Done.\n" << std::endl;
    }

    return input.leftCols( input.cols() - 1 );
}

template <Weight T>
[[nodiscard]] constexpr inline auto
NeuralNetwork<T>::backward_pass( const layer_t<T> & labels,
                                 const std::size_t  V ) noexcept {
    layer_t<T> expected_output{ layer_t<T>::Constant(
        labels.rows(), labels.cols() + 1, static_cast<T>( 1. ) ) };
    expected_output.leftCols( labels.cols() ) << labels;
    assert( expected_output.rows() == m_intermediate_state.back().rows()
            && expected_output.cols() == m_intermediate_state.back().cols() );
    auto layer_info_view{ std::views::zip( m_network, m_intermediate_state,
                                           m_activation_gradients )
                          | std::views::reverse | std::views::slide( 2 ) };
    if ( V )
        std::cout << "Backward pass:\n";

    for ( const auto & [i, layers] : layer_info_view | std::views::enumerate ) {
        // References to relevant layers
        const auto & cur_layer = layers[0];
        const auto & nxt_layer = layers[1];
        const auto & [nxt_weights, nxt_output, nxt_gradient] = nxt_layer;
        const auto & [cur_weights, cur_output, cur_gradient] = cur_layer;

        const auto d_cost{ m_cost_gradient( cur_output, expected_output ) };
        const auto d_cur_output{ cur_gradient( cur_output ) };
        const auto gradients{ d_cost.cwiseProduct( d_cur_output ) };
        const auto d_weights{ nxt_output.transpose() * gradients };
        // const auto new_weights{
        //     cur_weights - m_eta * d_weights.leftCols( d_weights.cols() - 1 )
        // };
        const auto new_weights{
            cur_weights - m_eta * d_weights.leftCols( cur_weights.cols() )
        };

        // TODO: Remove when working.
        if ( V ) {
            if ( V > 1 ) {
                std::cout << std::format( "cur_weights ({}, {})\n",
                                          cur_weights.rows(),
                                          cur_weights.cols() )
                          << cur_weights << "\n";
                std::cout << std::format( "new_weights ({}, {})\n",
                                          new_weights.rows(),
                                          new_weights.cols() )
                          << new_weights << "\n";
            }
            std::cout << "Weight diff:\n" << new_weights - cur_weights << "\n";
        }

        // Update current layer's weights
        cur_weights << new_weights;
        // Update input gradients for next layer
        // expected_output = gradients.leftCols( gradients.cols() - 1 )
        expected_output =
            gradients.leftCols( cur_weights.cols() ) * cur_weights.transpose();
    }
    if ( V )
        std::cout << "Done.\n" << std::endl;

    // std::cout << std::endl;
}
} // namespace neural
