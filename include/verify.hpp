#pragma once

#include <array>
#include <concepts>
#include <vector>

template <std::floating_point T, std::size_t N>
constexpr inline T MSE( const std::array<T, N> & predictions,
                        const std::array<T, N> & labels );
template <std::floating_point T>
T MSE( const std::vector<T> & predictions, const std::vector<T> & labels );