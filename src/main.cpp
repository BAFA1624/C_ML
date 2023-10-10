// #include "Eigen/Core"
#include "Eigen/Dense"
#include "neural_network.hpp"

#include <iostream>
#include <ranges>

using namespace neural;
using namespace neural::activation;

template <Weight T>
constexpr inline T
f( const T x ) {
    return std::sin( x );
}

template <Weight T>
constexpr inline layer_t<T>
gen_f_data( const T xmin, const T xmax, const std::size_t N ) {
    auto    result = layer_t<T>( N, 1 );
    const T dx{ ( xmax - xmin ) / static_cast<T>( N ) };
    for ( std::size_t i{ 0 }; i < N; ++i ) {
        const T xval{ xmin + dx * static_cast<T>( i ) };
        result.row( i ) << xval;
    }
    return result;
}

template <Weight T>
constexpr inline layer_t<T>
gen_f_labels( const layer_t<T> & inputs ) {
    auto result = layer_t<T>( inputs.rows(), 1 );
    for ( Eigen::Index i{ 0 }; i < inputs.rows(); ++i ) {
        result( i, 0 ) = f( inputs( i, 0 ) );
    }
    return result;
}

int
main() {
    std::cout << "Creating NeuralNetwork:" << std::endl;
    neural::NeuralNetwork<double> test(
        { 1, 10, 10, 10, 1 },
        { relu<double>, relu<double>, relu<double>, relu<double> },
        { d_relu<double>, d_relu<double>, d_relu<double>, d_relu<double> },
        cost::SSR<double>, cost::d_SSR<double>, 0.001 );
    std::cout << "Done." << std::endl;

    const auto input = gen_f_data<double>( 0., 10., 1000 );
    const auto labels = gen_f_labels<double>( input );

    std::cout << "input:\n" << input << std::endl;
    std::cout << "labels:\n" << labels << std::endl;
    std::cout << "output:\n" << test.forward_pass( input ) << std::endl;

    const std::size_t epochs{ 10000 };

    const auto initial_cost{ cost::SSR<double>( labels,
                                                test.forward_pass( input ) ) };

    std::cout << "Training..." << std::endl;
    for ( std::size_t i{ 0 }; i < epochs; ++i ) {
        const layer_t<double> output = test.forward_pass( input );
        test.backward_pass( labels );
    }
    std::cout << "Done." << std::endl;

    std::cout << "input:\n" << input << std::endl;
    std::cout << "output:\n" << test.forward_pass( input ) << std::endl;

    const auto final_cost{ cost::SSR<double>( labels,
                                              test.forward_pass( input ) ) };

    // const auto huh = layer_t<double>::Constant( 1, 3, 10000 );

    // std::cout << "huh:\n" << huh << std::endl;
    // const auto huh_output = test.forward_pass( huh );
    // std::cout << "huh_output:\n" << huh_output << std::endl;

    const auto initial_avg_cost{
        initial_cost.sum()
        / static_cast<double>( initial_cost.rows() * initial_cost.cols() )
    };
    const auto final_avg_cost{ final_cost.sum()
                               / static_cast<double>( final_cost.rows()
                                                      * final_cost.cols() ) };
    std::cout << std::format( "Avg. cost: {} -> {}", initial_avg_cost,
                              final_avg_cost )
              << std::endl;

    for ( const auto & [i, layer] : test.network() | std::views::enumerate ) {
        std::cout << i << "\n" << layer << std::endl;
    }

    const double x{ 0 }, dx{ 0.1 };
    for ( std::size_t i{ 0 }; i < 100; ++i ) {
        const double val{ x + static_cast<double>( i ) * dx };
        const auto   tmp{ layer_t<double>::Constant( 1, 1, val ) };
        std::cout << std::format( "sin({}) = {}, {}\n", val, std::sin( val ),
                                  test.forward_pass( tmp )( 0, 0 ) );
    }
}
