#pragma once

#include "neuron_utils.hpp"

namespace C_ML
{

namespace neuron
{

template <param_type T>
constexpr inline T
sigmoid( const T x ) {
    return std::exp( x ) / ( std::exp( x ) + 1.0 );
}

template <param_type T>
constexpr inline T
tanh( const T x ) {
    return std::tanh( x );
}

template <param_type T>
constexpr inline T
relu( const T x ) {
    if constexpr ( x > 0. ) {
        return x;
    }
    else {
        return 0.;
    }
}

template <param_type T>
constexpr inline T
leakrelu( const T x ) {
    if constexpr ( x > 0. ) {
        return x;
    }
    else {
        return 0.01 * x;
    }
}

template <param_type T>
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