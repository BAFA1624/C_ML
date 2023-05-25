#include "verify.hpp"

template <std::floating_point T>
T
MSE( const std::vector<T> & predictions, const std::vector<T> & labels ) {
    assert( predictions.size() == labels.size() );

    T MSE{ 0. };
    for ( std::uint64_t i{ 0 }; i < predictions.size(); ++i ) {
        MSE += ( predictions[i] - labels[i] ) * ( predictions[i] - labels[i] );
    }

    return MSE;
}

template <std::floating_point T, std::size_t N>
inline constexpr T
MSE( const std::array<T, N> & predictions, const std::array<T, N> & labels ) {
    T MSE{ 0. };
    for ( std::uint64_t i{ 0 }; i < N; ++i ) {
        MSE += ( predictions[i] - labels[i] ) * ( predictions[i] - labels[i] );
    }
    return MSE;
}
