#pragma once

#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <utility>
#include <variant>

#include "action.hpp"
#include "returns.hpp"
#include "reward.hpp"
#include "spec.hpp"
#include "step.hpp"
#include "transition.hpp"

template <typename T>
concept Float = std::is_floating_point<T>::value;

namespace environment {

using action::Action;
using action::ActionType;
using returns::Return;
using returns::ReturnType;
using reward::Reward;
using reward::RewardType;
using spec::CompositeArraySpec;
using state::State;
using step::Step;
using step::StepType;
using transition::Transition;
using transition::TransitionSequence;

template <StepType STEP_T, RewardType REWARD_T, ReturnType RETURN_T>
struct Environment {

  using EnvironmentType = Environment<STEP_T, REWARD_T, RETURN_T>;
  using StateType = typename STEP_T::StateType;
  using ActionSpace = typename STEP_T::ActionSpace;
  using ActionSpecType = typename ActionSpace::SpecType;
  using PrecisionType = typename STEP_T::PrecisionType;
  using StepType = STEP_T;
  using TransitionType = Transition<ActionSpace>;
  using RewardType = REWARD_T;
  using ReturnType = RETURN_T;

  template <std::size_t EPISODE_LENGTH>
  using EpisodeType = TransitionSequence<EPISODE_LENGTH, ActionSpace>;

  StateType state;

  Environment() = default;
  Environment(const StateType &s) : state(s) {}

  virtual void reset() = 0;
  virtual TransitionType step(const ActionSpace &action) {
    return TransitionType{state, action, StepType::step(state, action)};
  };
  void update(const TransitionType &t) { state = t.nextState; }
};

template <typename ENVIRON_T>
concept EnvironmentType = std::is_base_of_v<
    Environment<typename ENVIRON_T::StepType, typename ENVIRON_T::RewardType,
                typename ENVIRON_T::ReturnType>,
    ENVIRON_T>;

#define SINGLE_ARG(...) __VA_ARGS__
#define SETUP_TYPES(BASE_T)                                                    \
  using BaseType = BASE_T;                                                     \
  using StateType = typename BASE_T::StateType;                                \
  using ActionSpace = typename BASE_T::ActionSpace;                            \
  using ActionSpecType = typename BASE_T::ActionSpecType;                      \
  using PrecisionType = typename BASE_T::PrecisionType;                        \
  using StepType = typename BASE_T::StepType;                                  \
  using TransitionType = typename BASE_T::TransitionType;                      \
  using RewardType = typename BASE_T::RewardType;                              \
  using ReturnType = typename BASE_T::ReturnType;

} // namespace environment
