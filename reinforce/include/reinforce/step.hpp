#pragma once

#include <array>
#include <type_traits>

#include "reinforce/action.hpp"
#include "reinforce/spec.hpp"
#include "reinforce/state.hpp"

namespace step {

using action::ActionType;
using state::StateType;

template <ActionType ACTION0>
struct Step {
  using StateType = typename ACTION0::StateType;
  using PrecisionType = typename StateType::PrecisionType;
  using ActionSpace = ACTION0;
  using SpecType = typename ActionSpace::SpecType;

  // Calculate the next state given the current state and the action
  // For more complex models like markov transition states this can be overriden
  static StateType step(const StateType &state, const ActionSpace &action) { return action.step(state); };
};

template <typename STEP_T>
concept StepType = std::is_base_of_v<Step<typename STEP_T::ActionSpace>, STEP_T>;

} // namespace step
