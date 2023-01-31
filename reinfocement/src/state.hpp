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

template <Float TYPE_T, spec::CompositeArraySpecType OBSERVABLE_SPEC_T = spec::CompositeArraySpec<>,
          spec::CompositeArraySpecType HIDDEN_SPEC_T = spec::CompositeArraySpec<>>
struct State {
  using PrecisionType = TYPE_T;

  using ObservableSpecType = OBSERVABLE_SPEC_T;
  using HiddenSpecType = HIDDEN_SPEC_T;

  using ObservableDataType = typename ObservableSpecType::DataType;
  using HiddenDataType = typename HiddenSpecType::DataType;

  ObservableDataType observable = spec::default_spec_gen<ObservableSpecType>();
  HiddenDataType hidden = spec::default_spec_gen<HiddenSpecType>();

  State() = default;
  State(const ObservableDataType &o, const HiddenDataType &h) : observable(o), hidden(h){};

  virtual std::size_t hash() const { return observable.hash(); }

  friend bool operator==(const State &lhs, const State &rhs) { return lhs.observable == rhs.observable; }

  friend std::ostream &operator<<(std::ostream &os, const State &rhs) {
    os << "State(" << rhs.observable << ", " << rhs.hidden << ")";
    return os;
  }

  struct Hash {
    std::size_t operator()(const State &t) const { return t.hash(); }
  };
};

template <typename STATE_T>
concept StateType = std::is_base_of_v<
    State<typename STATE_T::PrecisionType, typename STATE_T::ObservableSpecType, typename STATE_T::HiddenSpecType>,
    STATE_T> ||
    std::is_same_v<
        State<typename STATE_T::PrecisionType, typename STATE_T::ObservableSpecType, typename STATE_T::HiddenSpecType>,
        STATE_T>;

} // namespace state
