#pragma once

#include "neural_util.hpp"

#include <algorithm>
#include <format>   // TODO: Remove
#include <iostream> // TODO: Remove
#include <numeric>
#include <ranges>
// #include <unsupported/Eigen/MatrixFunctions>

namespace neural
{

namespace activation
{

template <Weight T>
constexpr inline layer_t<T>
linear( const layer_t<T> & m ) {
    return m;
}

template <Weight T>
constexpr inline layer_t<T>
sigmoid( const layer_t<T> & m ) {
    return m.unaryExpr( []( const T x ) {
        return static_cast<T>( 1 ) / ( static_cast<T>( 1 ) + std::exp( -x ) );
    } );
}

template <Weight T>
constexpr inline layer_t<T>
d_sigmoid( const layer_t<T> & m ) {
    return m.unaryExpr(
        []( const T x ) { return x * ( static_cast<T>( 1 ) - x ); } );
}

template <Weight T>
constexpr inline layer_t<T>
relu( const layer_t<T> & m ) {
    return m.unaryExpr(
        []( const T x ) { return std::max( x, static_cast<T>( 0 ) ); } );
}

template <Weight T>
constexpr inline layer_t<T>
d_relu( const layer_t<T> & m ) {
    return m.unaryExpr( []( const T x ) {
        return static_cast<T>( ( x > static_cast<T>( 0 ) ) ? 1 : 0 );
    } );
}

template <Weight T>
constexpr inline layer_t<T>
softmax( const layer_t<T> & m ) {
    const auto m_exp = m.unaryExpr( []( const T x ) { return std::exp( x ); } );
    const auto m_exp_sum = m_exp.rowwise().sum();
    auto       result = layer_t<T>( m.rows(), m.cols() );
    for ( Eigen::Index i{ 0 }; i < m_exp_sum.rows(); ++i ) {
        result.row( i ) = m_exp.row( i ) / m_exp_sum( i, 0 );
    }

    return result;
}

/*template <Weight T>
constexpr inline Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
softmax( const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> & m ) {
    auto m_exp = m.unaryExpr( []( const T x ) { return std::exp( x ); } );
    auto row_sum = m_exp.rowwise().sum();

    for ( const auto & [i, pair] :
          std::views::zip( m_exp.rowwise(), row_sum.rowwise() )
              | std::views::enumerate ) {
        const auto & [row, sum] = pair;
        std::cout << std::format( "row {}: sum({}) -> {sum}\n", i, row, sum );
    }
    return m_exp;
}*/

/*template <Weight T>
constexpr inline std::vector<T>
softmax( const std::vector<T> & v ) {
    auto       result = std::vector<T>( v.size() );
    const auto exp_sum{ std::accumulate(
        v.cbegin(), v.cend(), static_cast<T>( 0 ),
        []( const auto & x, const auto & y ) { return x + std::exp( y ); } )
}; std::ranges::transform( v.cbegin(), v.cend(), result.begin(),
        [&exp_sum]( const auto & x ) { return std::exp( x ) / exp_sum; } );
    return result;
}*/

/*template <Weight T>
constexpr inline std::vector<T>
d_softmax( const std::vector<T> & v ) {
    auto result = std::vector<T>( v.size() );
    for ( const auto & [i, x] : v | std::views::enumerate ) {
        for ( const auto & [j, y] : v | std::views::enumerate ) {
            result[i] += ( i == j ? v[i] : static_cast<T>( 0 ) ) - v[i] *
v[j];
        }
    }
    return result;
}*/

/*template <Weight T>
constexpr inline layer_t<T>
d_softmax( const layer_t<T> & m ) {
    auto result = layer_t<T>( m.rows(), m.cols() );
    result.setZero();
    for ( Eigen::Index i{ 0 }; i < m.rows(); ++i ) {
        for ( Eigen::Index j{ 0 }; j < m.cols(); ++j ) { result }
    }
}*/

} // namespace activation

namespace cost
{

template <Weight T>
constexpr inline layer_t<T>
SSR( const layer_t<T> & predictions, const layer_t<T> & labels ) {
    return ( labels - predictions ).unaryExpr( []( const T x ) {
        return std::pow( x, 2 );
    } );
}

template <Weight T>
constexpr inline layer_t<T>
d_SSR( const layer_t<T> & predictions, const layer_t<T> & labels ) {
    return static_cast<T>( -2 ) * ( labels - predictions );
}

template <Weight T>
constexpr inline auto
MSE() {}

} // namespace  cost

} // namespace neural
