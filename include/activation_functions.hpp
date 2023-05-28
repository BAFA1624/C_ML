#pragma once

#include "neuron_base.hpp"

namespace C_ML
{

namespace neuron
{

template <neuron_weight T>
constexpr inline T
sigmoid( const T x ) {
    return std::exp( x ) / ( std::exp( x ) + 1.0 );
}

template <neuron_weight T>
constexpr inline T
relu( const T x ) {
    if constexpr ( x > 0. ) {
        return x;
    }
    else {
        return 0.;
    }
}

template <neuron_weight T>
constexpr inline T
leakrelu( const T x ) {
    if constexpr ( x > 0. ) {
        return x;
    }
    else {
        return 0.01 * x;
    }
}

template <neuron_weight T>
constexpr inline T
silu( const T x ) {
    if constexpr ( x > 0. ) {
        return x;
    }
    else {
        return sigmoid( x );
    }
}

}; // namespace neuron

}; // namespace C_ML