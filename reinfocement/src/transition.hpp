#pragma once

#include <array>
#include <type_traits>

#include "action.hpp"
#include "spec.hpp"
#include "state.hpp"
#include "step.hpp"

namespace transition {

using action::ActionType;
using step::Step;

// Terminal transitions precipitate a reset of the environment
enum class TransitionKind { NON_TERMINAL, TERMINAL };

template <ActionType ACTION0> struct Transition {
  using StateType = typename ACTION0::StateType;
  using PrecisionType = typename StateType::PrecisionType;
  using ActionSpace = ACTION0;
  using ActionSpecType = typename ActionSpace::SpecType;
  using StepType = Step<ActionSpace>;

  StateType state = StateType();
  ActionSpace action = ActionSpace();
  StateType nextState = StateType();
  TransitionKind kind = TransitionKind::NON_TERMINAL;

  bool operator==(const Transition &rhs) const {
    return state == rhs.state and action == rhs.action && nextState == rhs.nextState;
  }

  struct Hash {
    std::size_t operator()(const Transition &t) const { return t.state.hash() ^ t.action.hash() ^ t.nextState.hash(); }
  };

  bool isDone() const { return kind == TransitionKind::TERMINAL; }
};

template <typename TRANSITION_T>
concept isTransitionType = std::is_base_of_v<Transition<typename TRANSITION_T::ActionSpace>, TRANSITION_T>;

template <std::size_t SEQUENCE_LENGTH, ActionType ACTION0>
struct TransitionSequence : std::array<Transition<ACTION0>, SEQUENCE_LENGTH> {
  using StateType = typename ACTION0::StateType;
  using PrecisionType = typename StateType::PrecisionType;
  using ActionSpace = ACTION0;
  using ActionSpecType = typename ActionSpace::SpecType;
  using StepType = Step<ActionSpace>;
  using TransitionType = Transition<ActionSpace>;
  constexpr static std::size_t LENGTH = SEQUENCE_LENGTH;
};

} // namespace transition
