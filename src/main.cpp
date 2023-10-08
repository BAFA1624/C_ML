// #include "Eigen/Core"
#include "Eigen/Dense"
#include "neural_network.hpp"

#include <iostream>
#include <ranges>

using namespace neural;
using namespace neural::activation;

int
main() {
    std::cout << "Hello, world!" << std::endl;

    neural::NeuralNetwork<double> test(
        { 3, 5, 9 }, { sigmoid<double>, softmax<double> },
        { d_sigmoid<double>, d_sigmoid<double> }, cost::SSR<double> );
    //{ sigmoid<double>, sigmoid<double>, sigmoid<double>, softmax<double> },
    //{ d_sigmoid<double>, d_sigmoid<double>, d_sigmoid<double>,
    //  d_softmax<double> } );

    const auto shape = test.shape();

    Eigen::RowVector<double, Eigen::Dynamic> input( 1, 3 );
    input.setRandom();

    const layer_t<double> output = test.forward_pass( input );

    for ( const auto & [i, layer] :
          std::views::enumerate( test.intermediate_state() ) ) {
        std::cout << i << std::endl;
        std::cout << layer << "\n" << std::endl;
    }
    std::cout << "output:\n";
    std::cout << output << std::endl;

    auto label = layer_t<double>( output.rows(), output.cols() );
    label.setRandom();

    std::cout << "cost:\n" << cost::SSR( output, label ) << std::endl;

    std::cout << "cost gradient:\n" << -2. * ( label - output ) << std::endl;
}
