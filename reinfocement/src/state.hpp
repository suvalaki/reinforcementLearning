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

template <
    Float TYPE_T,
    spec::CompositeArraySpecType OBSERVABLE_SPEC_T = spec::CompositeArraySpec<>,
    spec::CompositeArraySpecType HIDDEN_SPEC_T = spec::CompositeArraySpec<>>
struct State {
  using PrecisionType = TYPE_T;

  using ObservableSpecType = OBSERVABLE_SPEC_T;
  using HiddenSpecType = HIDDEN_SPEC_T;

  using ObservableDataType = typename ObservableSpecType::DataType;
  using HiddenDataType = typename HiddenSpecType::DataType;

  ObservableDataType observable;
  HiddenDataType hidden;

  std::size_t hash() const { return hidden.hash(); }

  friend bool operator==(const State &lhs, const State &rhs) {
    return lhs.observable == rhs.observable;
  }
};

template <typename STATE_T>
concept StateType =
    std::is_base_of_v<State<typename STATE_T::PrecisionType>, STATE_T>;

} // namespace state
