#include "neural_network.hpp"

#include <iostream>

int
main() {
    std::cout << "Hello, world!" << std::endl;

    neural::NeuralNetwork<double> test( 10, 4, { 5, 4, 8 },
                                        std::time( nullptr ) );
}