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
    T                          m_eta_min;

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
        m_eta( eta ),
        m_eta_min( static_cast<T>( 0.001 ) * eta ) {
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

        std::uint64_t prev_layer_sz{ m_n_inputs };

        for ( std::uint64_t i{ 0 }; i < m_n_layers; ++i ) {
            const auto n_neurons{ m_neurons_per_layer[i] };
            m_network[i] = layer_t<T>::Random( prev_layer_sz + 1, n_neurons );
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

    constexpr inline layer_t<T>
    forward_pass( const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &
                      inputs ) noexcept;
    constexpr inline void backward_pass( const layer_t<T> & labels ) noexcept;
    constexpr inline void train( const layer_t<T> & labels,
                                 const layer_t<T> & samples,
                                 const std::size_t  N ) noexcept;
};

template <Weight T>
constexpr inline layer_t<T>
NeuralNetwork<T>::forward_pass( const layer_t<T> & inputs ) noexcept {
    // Input matrix w/ extra col. for bias inputs (1)
    layer_t<T> input{ layer_t<T>::Constant( inputs.rows(), inputs.cols() + 1,
                                            static_cast<T>( 1. ) ) };
    assert( input.cols() == m_network.front().rows() );
    input.leftCols( inputs.cols() ) << inputs;
    m_intermediate_state[0] = input;
    const auto view = std::views::zip( m_network, m_intermediate_state,
                                       m_activation_functions )
                      | std::views::drop( 1 );
    for ( const auto & [i, layers] : view | std::views::enumerate ) {
        auto & [network_layer, output_layer, act_func] = layers;

        output_layer = layer_t<T>::Constant(
            input.rows(), network_layer.cols() + 1, static_cast<T>( 1. ) );

        output_layer.leftCols( output_layer.cols() - 1 )
            << act_func( input * network_layer );

        input = output_layer;
    }

    return input.leftCols( input.cols() - 1 );
}

template <Weight T>
constexpr inline void
NeuralNetwork<T>::backward_pass( const layer_t<T> & labels ) noexcept {
    layer_t<T> expected_output{ layer_t<T>::Constant(
        labels.rows(), labels.cols() + 1, static_cast<T>( 1. ) ) };
    expected_output.leftCols( labels.cols() ) << labels;
    assert( expected_output.rows() == m_intermediate_state.back().rows()
            && expected_output.cols() == m_intermediate_state.back().cols() );
    auto layer_info_view{ std::views::zip( m_network, m_intermediate_state,
                                           m_activation_gradients )
                          | std::views::reverse | std::views::slide( 2 ) };

    // Calculate initial cost gradient
    auto d_cost{ m_cost_gradient( m_intermediate_state.back(),
                                  expected_output ) };
    for ( const auto & [i, layers] : layer_info_view | std::views::enumerate ) {
        // References to relevant layers
        const auto & cur_layer = layers[0];
        const auto & nxt_layer = layers[1];
        const auto & [nxt_weights, nxt_output, nxt_gradient] = nxt_layer;
        const auto & [cur_weights, cur_output, cur_gradient] = cur_layer;

        const auto d_cur_output{ cur_gradient( cur_output ) };
        const auto gradients{ d_cost.cwiseProduct( d_cur_output ) };
        const auto d_weights{ nxt_output.transpose() * gradients };
        const auto new_weights{
            cur_weights - m_eta * d_weights.leftCols( cur_weights.cols() )
        };

        // Update current layer's weights
        cur_weights << new_weights;
        // Update input gradients for next layer
        // expected_output = gradients.leftCols( gradients.cols() - 1 )
        d_cost =
            gradients.leftCols( cur_weights.cols() ) * cur_weights.transpose();
    }
}

template <Weight T>
constexpr inline void
NeuralNetwork<T>::train( const layer_t<T> & labels, const layer_t<T> & samples,
                         const std::size_t N ) noexcept {
    for ( std::size_t i{ 0 }; i < N; ++i ) {
        forward_pass( samples );
        backward_pass( labels );
    }
}


} // namespace neural
