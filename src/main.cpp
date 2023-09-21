#include "neural_network.hpp"

#include <iostream>

int
main() {
    std::cout << "Hello, world!" << std::endl;

    neural::NeuralNetwork<double> test(
        { 3, 4, 5, 6, 3 },
        { neural::sigmoid<double>, neural::sigmoid<double>,
          neural::sigmoid<double>, neural::sigmoid<double> },
        { neural::d_sigmoid<double>, neural::d_sigmoid<double>,
          neural::d_sigmoid<double>, neural::d_sigmoid<double> } );

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
    test.backward_pass( {} );
}
