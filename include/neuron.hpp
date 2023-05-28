#pragma once

#include "activation_functions.hpp"
#include "neuron_base.hpp"


namespace C_ML
{

namespace neuron
{

template <std::floating_point T, init_t Init>
constexpr inline void
initialise( T & x ) {
    if constexpr ( Init == init_t::random ) {
        x = static_cast<T>( std::rand() ) / static_cast<T>( RAND_MAX );
    }
}

template <std::floating_point T, init_t Init>
constexpr inline void
initialise( std::complex<T> & z ) {
    if constexpr ( Init == init_t::random ) {
        z = { static_cast<T>( std::rand() ) / static_cast<T>( RAND_MAX ),
              static_cast<T>( std::rand() ) / static_cast<T>( RAND_MAX ) };
    }
}


template <neuron_weight T, init_t Init = init_t::random>
class Neuron
{
    private:
    T                                       m_bias;
    std::vector<T>                          m_weights;
    std::vector<std::shared_ptr<Neuron<T>>> m_inputs;
    activation_func<T>                      m_activation;

    public:
    Neuron( const activation_func<T>                        f,
            const std::vector<std::shared_ptr<Neuron<T>>> & inputs ) :
        m_inputs( inputs ), m_activation( f ) {
        m_weights.resize( inputs.size() );
        if constexpr ( !is_complex<T>::value ) {
            initialise<T, Init>( m_bias );
            for ( auto & weight : m_weights ) { initialise<T, Init>( weight ); }
        }
        else {
            initialise<typename T::value_type, Init>( m_bias );
            for ( auto & weight : m_weights ) {
                initialise<typename T::value_type, Init>( weight );
            }
        }
    }
    virtual ~Neuron() = default;

    [[nodiscard]] constexpr auto weights() const noexcept { return m_weights; }
    [[nodiscard]] constexpr auto bias() const noexcept { return m_bias; }

    [[nodiscard]] virtual T cost() const noexcept {
        return m_activation(
            m_bias
            + std::accumulate( m_inputs.cbegin(), m_inputs.cend(),
                               static_cast<T>( 0 ),
                               []( const auto b, const auto & n ) {
                                   return b + n->cost();
                               } ) );
    };
};

template <neuron_weight T, init_t Init = init_t::random>
class OutputNeuron final : public Neuron<T>
{};

}; // namespace neuron

}; // namespace C_ML