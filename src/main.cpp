#include "neural_network.hpp"

#include <iostream>

int
main() {
    std::cout << "Hello, world!" << std::endl;

    neural::NeuralNetwork<double> test(
        { 3, 4, 5, 6, 3 },
        { neural::sigmoid<double>, neural::sigmoid<double>,
          neural::sigmoid<double>, neural::sigmoid<double> },
        std::time( nullptr ) );

    const auto shape = test.shape();

    std::cout << "Network shape:" << std::endl;
    for ( const auto & x : shape ) { std::cout << x << std::endl; }

    const auto res = test.forward(
        std::vector<double>{ 0., 1., 2., 3., 4., 5., 6., 7., 8., 9. } );
    std::cout << "Testing forwarding:" << std::endl;
    for ( const auto & x : res ) { std::cout << x << std::endl; }
}
