#pragma once


#include "neuron_base.hpp"


namespace C_ML
{

template <param_type T>
T
MSE( const std::vector<T> & predictions, const std::vector<T> & labels ) {
    assert( predictions.size() == labels.size() );

    T MSE{ 0. };
    for ( std::uint64_t i{ 0 }; i < predictions.size(); ++i ) {
        MSE += ( predictions[i] - labels[i] ) * ( predictions[i] - labels[i] );
    }

    return MSE;
}

template <param_type T, std::size_t N>
inline constexpr T
MSE( const std::array<T, N> & predictions, const std::array<T, N> & labels ) {
    T MSE{ 0. };
    for ( std::uint64_t i{ 0 }; i < N; ++i ) {
        std::cout << "Prediction: " << predictions[i]
                  << ", label: " << labels[i] << std::endl;
        MSE += ( predictions[i] - labels[i] ) * ( predictions[i] - labels[i] );
    }
    return MSE;
}

}; // namespace C_ML