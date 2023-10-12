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
        { sigmoid<double>, sigmoid<double>, sigmoid<double>, sigmoid<double> },
        { d_sigmoid<double>, d_sigmoid<double>, d_sigmoid<double>,
          d_sigmoid<double> },
        cost::SSR<double>, cost::d_SSR<double>, 0.005 );
    std::cout << "Done." << std::endl;

    const auto input = gen_f_data<double>( 0., 3.2, 1000 );
    const auto labels = gen_f_labels<double>( input );

    // const std::size_t epochs{ 10000 };
    const std::size_t epochs{ 100 };

    std::cout << "Training..." << std::endl;
    for ( std::size_t i{ 0 }; i < epochs; ++i ) {
        std::cout << "epoch " << i + 1 << ":\n";

        const auto initial_cost{ cost::SSR<double>(
            labels, test.forward_pass( input ) ) };

        const layer_t<double> output = test.forward_pass( input, 2 );
        test.backward_pass( labels, 2 );

        const auto final_cost{ cost::SSR<double>(
            labels, test.forward_pass( input ) ) };

        const auto initial_avg_cost{
            initial_cost.sum()
            / static_cast<double>( initial_cost.rows() * initial_cost.cols() )
        };
        const auto final_avg_cost{
            final_cost.sum()
            / static_cast<double>( final_cost.rows() * final_cost.cols() )
        };
        std::cout << std::format( "Avg. cost: {} -> {}", initial_avg_cost,
                                  final_avg_cost )
                  << std::endl;
    }
    std::cout << "Done." << std::endl;

    std::cout << "f(x) = 5 * sin(x)\t\t\tprediction\tcost:\n";
    const double x{ 0 }, dx{ 0.3 };
    for ( std::size_t i{ 0 }; i < 20; ++i ) {
        const double val{ x + static_cast<double>( i ) * dx };
        const auto   tmp{ layer_t<double>::Constant( 1, 1, val ) };
        const auto   cost{ cost::SSR<double>( test.forward_pass( tmp ), tmp ) };
        std::cout << std::format( "f({}) = {}\t\t\t{}\t(cost = {})\n", val,
                                  f( val ), test.forward_pass( tmp )( 0, 0 ),
                                  cost( 0, 0 ) );
    }
}
