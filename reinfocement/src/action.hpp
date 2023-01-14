#pragma once

#include <type_traits>

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

  // Add type which adheres to the spec
  Action(const typename T::DataType &d) : T::DataType(d) {}

  // When an action modifies the state space impliment anew
  virtual StateType step(const StateType &state) const { return state; }
};

template <typename ACTION_T>
concept ActionType = std::is_base_of_v<
    Action<typename ACTION_T::StateType, typename ACTION_T::SpecType>,
    ACTION_T>;

// TODO : CONCEPT TO REQUIRE ALL ACTIONS TO BE OVER THE SAME BASE

} // namespace action
