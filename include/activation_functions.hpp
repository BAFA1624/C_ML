#pragma once

#include <cmath>
#include <concepts>

template <std::floating_point T>
constexpr inline T
sigmoid( const T x ) {
    return std::exp( x ) / ( std::exp( x ) + 1.0 );
}

template <std::floating_point T>
constexpr inline T
relu( const T x ) {
    if constexpr ( x > 0. ) {
        return x;
    }
    else {
        return 0.;
    }
}

template <std::floating_point T>
constexpr inline T
leakrelu( const T x ) {
    if constexpr ( x > 0. ) {
        return x;
    }
    else {
        return 0.01 * x;
    }
}

template <std::floating_point T>
constexpr inline T
silu( const T x ) {
    if constexpr ( x > 0. ) {
        return x;
    }
    else {
        return sigmoid( x );
    }
}
