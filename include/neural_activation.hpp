#pragma once

#include "neural_util.hpp"

namespace neural
{

template <weight_type T>
constexpr inline T
linear( const T x ) {
    return x;
}

template <weight_type T>
constexpr inline T
sigmoid( const T x ) {
    return static_cast<T>( 1 ) / ( 1 + std::exp( -x ) );
}

} // namespace neural
