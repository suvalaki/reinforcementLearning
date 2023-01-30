#pragma once

#include <boost/functional/hash.hpp>
#include <string>
#include <type_traits>
#include <unordered_map>

#include <xtensor/xarray.hpp>
#include <xtensor/xstrided_view.hpp>

#include "spec.hpp"
#include "state.hpp"

namespace action {

using typename state::StateType;

template <StateType STATE_T, spec::CompositeArraySpecType T>
struct Action : T::DataType {
  // using typename T::DataType;
  using SpecType = T;
  using StateType = STATE_T;
  using PrecisionType = typename StateType::PrecisionType;
  using DataType = T::DataType;

  // Add type which adheres to the spec
  Action() = default;
  Action(const typename T::DataType &d) : T::DataType(d) {}

  // When an action modifies the state space impliment anew
  virtual StateType step(const StateType &state) const { return state; }

  bool operator==(const Action &rhs) {
    return static_cast<const typename T::DataType &>(*this) ==
           static_cast<const typename T::DataType &>(rhs);
  }

  // create a single long number which is the same as the string concat of all
  // the elements in the data array
  virtual std::size_t hash() const {
    return static_cast<const typename T::DataType &>(*this).hash();
  }

  struct Hash {
    std::size_t operator()(const Action &t) const { return t.hash(); }
  };
};

template <typename ACTION_T>
concept ActionType = std::is_base_of_v<
    Action<typename ACTION_T::StateType, typename ACTION_T::SpecType>,
    ACTION_T>;

// TODO : CONCEPT TO REQUIRE ALL ACTIONS TO BE OVER THE SAME BASE

} // namespace action
