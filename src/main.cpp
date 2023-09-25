// #include "Eigen/Core"
#include "Eigen/Dense"
#include "neural_network.hpp"

#include <iostream>

using namespace neural;
using namespace neural::activation;

int
main() {
    std::cout << "Hello, world!" << std::endl;

    neural::NeuralNetwork<double> test(
        { 3, 4, 5, 6, 3 },
        { sigmoid<double>, sigmoid<double>, sigmoid<double>, softmax<double> },
        { d_sigmoid<double>, d_sigmoid<double>, d_sigmoid<double>,
          d_softmax<double> } );

    const auto shape = test.shape();

    test.forward_pass( std::vector<double>{ 0., 1., 2. } );

    const auto & internal_outputs = test.intermediate_state();
    for ( const auto & [i, layer] : internal_outputs | std::views::enumerate ) {
        for ( std::cout << "Layer (size = " << layer.size() << "): " << i
                        << "\n\t";
              const auto & output : layer ) {
            std::cout << output << ' ';
        }
        std::cout << std::endl;
    }
    // test.backward_pass( {} );
}
