#pragma once

#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <type_traits>
#include <unordered_set>
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
using transition::TransitionKind;
using transition::TransitionSequence;

template <StepType STEP_T, RewardType REWARD_T, ReturnType RETURN_T> struct Environment {

  using EnvironmentType = Environment<STEP_T, REWARD_T, RETURN_T>;
  using StateType = typename STEP_T::StateType;
  using ActionSpace = typename STEP_T::ActionSpace;
  using ActionSpecType = typename ActionSpace::SpecType;
  using PrecisionType = typename STEP_T::PrecisionType;
  using StepType = STEP_T;
  using TransitionType = Transition<ActionSpace>;
  using RewardType = REWARD_T;
  using ReturnType = RETURN_T;

  template <std::size_t EPISODE_LENGTH> using EpisodeType = TransitionSequence<EPISODE_LENGTH, ActionSpace>;

  StateType state;

  Environment() = default;
  Environment(const StateType &s) : state(s) {}

  virtual StateType reset() = 0;
  virtual TransitionType step(const ActionSpace &action) {
    return TransitionType{state, action, StepType::step(state, action)};
  };
  void update(const TransitionType &t) {
    if (t.isDone()) {
      reset();
    } else {
      state = t.nextState;
    }
  }
  virtual StateType getNullState() const = 0;
};

template <typename ENVIRON_T>
concept EnvironmentType = std::is_base_of_v<
    Environment<typename ENVIRON_T::StepType, typename ENVIRON_T::RewardType, typename ENVIRON_T::ReturnType>,
    ENVIRON_T>;

#define SINGLE_ARG(...) __VA_ARGS__
#define SETUP_TYPES(BASE_T)                                                                                            \
  using BaseType = BASE_T;                                                                                             \
  using StateType = typename BASE_T::StateType;                                                                        \
  using ActionSpace = typename BASE_T::ActionSpace;                                                                    \
  using ActionSpecType = typename BASE_T::ActionSpecType;                                                              \
  using PrecisionType = typename BASE_T::PrecisionType;                                                                \
  using StepType = typename BASE_T::StepType;                                                                          \
  using TransitionType = typename BASE_T::TransitionType;                                                              \
  using RewardType = typename BASE_T::RewardType;                                                                      \
  using ReturnType = typename BASE_T::ReturnType;

#define SETUP_TYPES_FROM_ENVIRON(ENVIRON_T)                                                                            \
  using EnvironmentType = ENVIRON_T;                                                                                   \
  using StateType = typename ENVIRON_T::StateType;                                                                     \
  using ActionSpace = typename ENVIRON_T::ActionSpace;                                                                 \
  using ActionSpecType = typename ENVIRON_T::ActionSpecType;                                                           \
  using PrecisionType = typename ENVIRON_T::PrecisionType;                                                             \
  using StepType = typename ENVIRON_T::StepType;                                                                       \
  using TransitionType = typename ENVIRON_T::TransitionType;                                                           \
  using RewardType = typename ENVIRON_T::RewardType;                                                                   \
  using ReturnType = typename ENVIRON_T::ReturnType;

#define SETUP_TYPES_W_ENVIRON(BASE_T, ENVIRON_T)                                                                       \
  using BaseType = BASE_T;                                                                                             \
  using EnvironmentType = ENVIRON_T;                                                                                   \
  using StateType = typename BASE_T::StateType;                                                                        \
  using ActionSpace = typename BASE_T::ActionSpace;                                                                    \
  using ActionSpecType = typename BASE_T::ActionSpecType;                                                              \
  using PrecisionType = typename BASE_T::PrecisionType;                                                                \
  using StepType = typename BASE_T::StepType;                                                                          \
  using TransitionType = typename BASE_T::TransitionType;                                                              \
  using RewardType = typename BASE_T::RewardType;                                                                      \
  using ReturnType = typename BASE_T::ReturnType;

template <StepType STEP_T, RewardType REWARD_T, ReturnType RETURN_T, std::size_t N_STATES, std::size_t N_ACTIONS>
struct FiniteEnvironment : Environment<STEP_T, REWARD_T, RETURN_T> {

  SETUP_TYPES(SINGLE_ARG(Environment<STEP_T, REWARD_T, RETURN_T>));
  using EnvironmentType = FiniteEnvironment;
  using BaseType::BaseType;

  std::random_device rd;
  mutable std::mt19937 gen{rd()};

  constexpr static std::size_t nStates = N_STATES;
  constexpr static std::size_t nActions = N_ACTIONS;

  virtual StateType stateFromIndex(std::size_t) const = 0;
  virtual ActionSpace actionFromIndex(std::size_t) const = 0;

  virtual std::unordered_set<StateType, typename StateType::Hash> getAllPossibleStates() const = 0;
  virtual std::unordered_set<ActionSpace, typename ActionSpace::Hash> getAllPossibleActions() const = 0;
  virtual std::unordered_set<StateType, typename StateType::Hash> getReachableStates(const StateType &s,
                                                                                     const ActionSpace &a) const {
    return getAllPossibleStates();
  }
  virtual std::unordered_set<ActionSpace, typename ActionSpace::Hash> getReachableActions(const StateType &s) const {
    return getAllPossibleActions();
  }

  StateType randomState() const {
    const auto probabilities = std::vector<double>(nStates, 1.0 / N_STATES);
    std::discrete_distribution<size_t> distribution(probabilities.begin(), probabilities.end());
    return stateFromIndex(distribution(gen));
  }

  ActionSpace randomAction() const {
    const auto probabilities = std::vector<double>(nActions, 1.0 / N_ACTIONS);
    std::discrete_distribution<size_t> distribution(probabilities.begin(), probabilities.end());
    return actionFromIndex(distribution(gen));
  }
  virtual ActionSpace randomAction(const StateType &s) const {
    const auto admissibleActions = getReachableActions(s);
    const auto probabilities = std::vector<double>(admissibleActions.size(), 1.0 / admissibleActions.size());
    std::discrete_distribution<size_t> distribution(probabilities.begin(), probabilities.end());
    return *std::next(admissibleActions.begin(), distribution(gen));
  }
};

template <typename ENVIRON_T>
concept FiniteEnvironmentType = std::is_base_of_v<FiniteEnvironment<typename ENVIRON_T::StepType,
                                                                    typename ENVIRON_T::RewardType,
                                                                    typename ENVIRON_T::ReturnType,
                                                                    ENVIRON_T::nStates,
                                                                    ENVIRON_T::nActions>,
                                                  ENVIRON_T>;

template <typename T>
concept FullyKnownFiniteActionStateEnvironment = FiniteEnvironmentType<T> && requires(T t) {
  { t.getAllPossibleStates() } -> std::same_as<std::unordered_set<typename T::StateType, typename T::StateType::Hash>>;
  {
    t.getAllPossibleActions()
    } -> std::same_as<std::unordered_set<typename T::ActionSpace, typename T::ActionSpace::Hash>>;
};

template <typename T>
concept FullyKnownFiniteStateEnvironment = FiniteEnvironmentType<T> && requires(T t) {
  { t.getAllPossibleStates() } -> std::same_as<std::unordered_set<typename T::StateType, typename T::StateType::Hash>>;
};

template <typename T>
concept FullyKnownConditionalStateActionEnvironment = FiniteEnvironmentType<T> && requires(T t) {
  { t.getAllPossibleStates() } -> std::same_as<std::unordered_set<typename T::StateType, typename T::StateType::Hash>>;
  {
    t.getReachableActions(std::declval<const typename T::StateType &>())
    } -> std::same_as<std::unordered_set<typename T::ActionSpace, typename T::ActionSpace::Hash>>;
};

} // namespace environment
