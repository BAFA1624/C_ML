#pragma once

#include "neural_util.hpp"

#include <algorithm>
#include <numeric>
#include <ranges>

namespace neural
{

namespace activation
{

template <Weight T>
constexpr inline std::vector<T>
linear( const std::vector<T> & v ) {
    return v;
}

template <Weight T>
constexpr inline std::vector<T>
sigmoid( const std::vector<T> & v ) {
    auto result = std::vector<T>( v.size() );
    std::ranges::transform(
        v.cbegin(), v.cend(), result.begin(), []( const auto & x ) {
            return static_cast<T>( 1 )
                   / ( static_cast<T>( 1 ) + std::exp( -x ) );
        } );
    return result;
}

template <Weight T>
constexpr inline std::vector<T>
d_sigmoid( const std::vector<T> & v ) {
    auto result = std::vector<T>( v.size() );
    std::ranges::transform(
        v.cbegin(), v.cend(), result.begin(),
        []( const auto & x ) { return x * ( static_cast<T>( 1 ) - x ); } );
    return result;
}

template <Weight T>
constexpr inline std::vector<T>
softmax( const std::vector<T> & v ) {
    auto       result = std::vector<T>( v.size() );
    const auto exp_sum{ std::accumulate(
        v.cbegin(), v.cend(), static_cast<T>( 0 ),
        []( const auto & x, const auto & y ) { return x + std::exp( y ); } ) };
    std::ranges::transform(
        v.cbegin(), v.cend(), result.begin(),
        [&exp_sum]( const auto & x ) { return std::exp( x ) / exp_sum; } );
    return result;
}

template <Weight T>
constexpr inline std::vector<T>
d_softmax( const std::vector<T> & v ) {
    auto result = std::vector<T>( v.size() );
    for ( const auto & [i, x] : v | std::views::enumerate ) {
        for ( const auto & [j, y] : v | std::views::enumerate ) {
            result[i] += ( i == j ? v[i] : static_cast<T>( 0 ) ) - v[i] * v[j];
        }
    }
    return result;
}

} // namespace activation

namespace cost
{

template <Weight T>
constexpr inline auto
MSE() {}

} // namespace  cost

} // namespace neural
