#include "neuron.hpp"
#include "verify.hpp"

#include <array>
#include <iostream>
#include <vector>

std::vector<std::array<double, 11>> data = {
    { 0., 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0 },
    { 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0 }
};

int
main() {
    std::array<double, 11> test_predictions;
    test_predictions.fill( 1. );

    std::cout << C_ML::MSE( test_predictions, data[1] ) << std::endl;

    C_ML::neuron::Neuron<double> test( 0.5, C_ML::neuron::sigmoid<double> );
}