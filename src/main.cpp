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
    return 5 * std::sin( x );
}

template <Weight T>
constexpr inline layer_t<T>
gen_f_data( const T xmin, const T xmax, const std::size_t N ) {
    auto    result = layer_t<T>( N, 1 );
    const T dx{ ( xmax - xmin ) / static_cast<T>( N ) };
    for ( std::size_t i{ 0 }; i < N; ++i ) {
        const T xval{ xmin + dx * static_cast<T>( i ) };
        result.row( static_cast<Eigen::Index>( i ) ) << xval;
    }
    return result;
}

template <Weight T>
constexpr inline T
OR( const T x1, const T x2 ) {
    return ( x1 || x2 ) ? 1. : 0.;
}

template <Weight T>
constexpr inline T
AND( const T x1, const T x2 ) {
    return ( x1 && x2 ) ? 1. : 0.;
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
    neural::NeuralNetwork<double> test( { 2, 1 }, { tanh<double> },
                                        { d_tanh<double> }, cost::SSR<double>,
                                        cost::d_SSR<double>, 0.015 );
    std::cout << "Done." << std::endl;

    const auto f_input = gen_f_data<double>( 0., 1., 10 );
    const auto f_labels = gen_f_labels<double>( f_input );

    auto OR_samples{ layer_t<double>( 4, 2 ) };
    OR_samples << 0, 0, 0, 1, 1, 0, 1, 1;
    const auto AND_samples{ OR_samples };
    const auto NAND_samples{ OR_samples };
    const auto NOR_samples{ OR_samples };
    const auto XOR_samples{ OR_samples };

    auto OR_labels{ layer_t<double>( 4, 1 ) };
    auto AND_labels{ OR_labels };
    auto NAND_labels{ OR_labels };
    auto NOR_labels{ OR_labels };
    auto XOR_labels{ OR_labels };
    OR_labels << 0, 1, 1, 1;
    AND_labels << 0, 0, 0, 1;
    NAND_labels << 1, 1, 1, 0;
    NOR_labels << 1, 0, 0, 0;
    XOR_labels << 0, 1, 1, 0;

    const auto func_name{ "NAND" };
    const auto inputs{ NAND_samples };
    const auto labels{ NAND_labels };

    const auto initial_cost{ cost::SSR<double>( labels,
                                                test.forward_pass( inputs ) ) };
    const auto initial_avg_cost{
        initial_cost.sum()
        / static_cast<double>( initial_cost.rows() * initial_cost.cols() )
    };

    const std::size_t epochs{ 1000 };

    std::cout << "Training for " << epochs << " epochs..." << std::endl;
    test.train( labels, inputs, epochs, 5 );
    std::cout << "Done." << std::endl;

    const auto final_cost{ cost::SSR<double>( labels,
                                              test.forward_pass( inputs ) ) };
    const auto final_avg_cost{ final_cost.sum()
                               / static_cast<double>( final_cost.rows()
                                                      * final_cost.cols() ) };
    std::cout << std::format( "Avg. cost change: {} -> {}", initial_avg_cost,
                              final_avg_cost )
              << std::endl;

    std::cout << std::format( "{}:", func_name ) << std::endl;
    for ( Eigen::Index i{ 0 }; i < inputs.rows(); ++i ) {
        std::cout << std::format(
            "{} || {} -> {}\t({})\n", inputs( i, 0 ), inputs( i, 1 ),
            labels( i, 0 ), test.forward_pass( inputs.row( i ) )( 0, 0 ) );
    }
}
