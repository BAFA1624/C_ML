#pragma once

#include "neural_util.hpp"

namespace neural
{

template <Weight T>
constexpr inline T
linear( const T x ) {
    return x;
}

template <Weight T>
constexpr inline T
sigmoid( const T x ) {
    return static_cast<T>( 1 ) / ( static_cast<T>( 1 ) + std::exp( -x ) );
}

template <Weight T>
constexpr inline T
d_sigmoid( const T x ) {
    return x * ( static_cast<T>( 1 ) - x );
}

} // namespace neural
