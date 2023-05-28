#include "neuron.hpp"
#include "verify.hpp"

#include <array>
#include <iostream>
#include <vector>

std::vector<std::array<double, 11>> data = {
    { 0., 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0 },
    { 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0 }
};
std::vector<std::array<std::complex<double>, 11>> complex_data = {
    { std::complex<double>{ 0.0, 0.0 }, std::complex<double>{ 1.0, -1.0 },
      std::complex<double>{ 2.0, -2.0 }, std::complex<double>{ 3.0, -3.0 },
      std::complex<double>{ 4.0, -4.0 }, std::complex<double>{ 5.0, -5.0 },
      std::complex<double>{ 6.0, -6.0 }, std::complex<double>{ 7.0, -7.0 },
      std::complex<double>{ 8.0, -8.0 }, std::complex<double>{ 9.0, -9.0 },
      std::complex<double>{ 10.0, -10.0 } },
    { std::complex<double>{ 0.0, 0.0 }, std::complex<double>{ 0.0, 0.0 },
      std::complex<double>{ 0.0, 0.0 }, std::complex<double>{ 0.0, 0.0 },
      std::complex<double>{ 0.0, 0.0 }, std::complex<double>{ 0.0, 0.0 },
      std::complex<double>{ 0.0, 0.0 }, std::complex<double>{ 0.0, 0.0 },
      std::complex<double>{ 0.0, 0.0 }, std::complex<double>{ 0.0, 0.0 },
      std::complex<double>{ 0.0, 0.0 } }
};

int
main() {
    std::srand( time( NULL ) );
    for ( auto i{ 0 }; i < std::rand(); ++i ) { std::rand(); }

    using T = std::complex<double>;

    std::array<T, 11> test_predictions;
    test_predictions.fill( { 1., 1. } );

    std::cout << C_ML::MSE( test_predictions, complex_data[0] ) << std::endl;

    C_ML::neuron::Neuron<T> test( C_ML::neuron::sigmoid<T>, {} );
    std::cout << test.cost() << std::endl;
}