#pragma once

#include <cmath>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <utility>
#include <variant>

#include <array>

#include "spec.hpp"

namespace state {

using spec::Float;

template <Float TYPE_T> struct State { using PrecisionType = TYPE_T; };

template <typename STATE_T>
concept StateType =
    std::is_base_of_v<State<typename STATE_T::PrecisionType>, STATE_T>;

} // namespace state
