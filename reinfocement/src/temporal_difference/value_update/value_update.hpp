#pragma once

#include <unordered_map>
#include <vector>

#include "monte_carlo/episode.hpp"
#include "policy/distribution_policy.hpp"
#include "policy/finite/greedy_policy.hpp"
#include "policy/finite/value_policy.hpp"
#include "policy/objectives/finite_value_function.hpp"
#include "policy/value.hpp"

namespace temporal_difference {

auto temporal_differenc_error(
    const auto &valueAtStateT, const auto &valueAtStateT1, const auto &reward, const auto &discountRate) {
  return reward + discountRate * valueAtStateT1 - valueAtStateT;
}

/// @brief Value update interface. Value updates are responsible for modifying the value function during
/// episodic monte carlo value estimation. They may have internal state - such is a list of returns or
/// importance sampling rations.
template <typename T>
concept isTDValueUpdater = requires(T t) {

  typename T::ValueFunctionType;
  typename T::EnvironmentType;
  typename T::KeyType;
  typename T::KeyMaker;
  typename T::ValueType;
  typename T::PrecisionType;
  typename T::StateType;
  typename T::ActionSpace;
  // typename T::ReturnsContainer;

  // Initialise both the environment and the value function. It may be uneccessary to initialise the value
  // function because it will be automatically initialised on the first update.
  t.initialize(std::declval<typename T::EnvironmentType &>(), std::declval<typename T::ValueFunctionType &>());

  // Using the newly updated state of the Updater, update the value function.
  // t.updateValue(
  //     std::declval<typename T::ValueFunctionType &>(),
  //     std::declval<policy::FiniteGreedyPolicy<typename T::KeyMaker, typename T::ValueType> &>(),
  //     std::declval<policy::FiniteGreedyPolicy<typename T::KeyMaker, typename T::ValueType> &>(),
  //     std::declval<typename T::EnvironmentType &>(),
  //     std::declval<const KeyType &>(),
  //     std::declval<const KeyType &>(),
  //     std::declval<const PrecisionType &>(),
  //     std::declval<const PrecisionType &>());
  // t.updateValue(std::declval<typename T::ValueFunctionType &>(),
  //               std::declval<policy::FiniteGreedyPolicy<typename T::KeyMaker, typename T::ValueType> &>(),
  //               std::declval<policy::FiniteGreedyPolicy<typename T::KeyMaker, typename T::ValueType> &>(),
  //               std::declval<typename T::EnvironmentType &>(),
  //               std::declval<const KeyType &>(),
  //               std::declval<const KeyType &>(),
  //               std::declval<const PrecisionType &>(),
  //               std::declval<const PrecisionType &>());
  // t.update(
  //     std::declval<typename T::ValueFunctionType &>(),
  //     std::declval<policy::FiniteGreedyPolicy<typename T::KeyMaker, typename T::ValueType> &>(),
  //     std::declval<policy::FiniteGreedyPolicy<typename T::KeyMaker, typename T::ValueType> &>(),
  //     std::declval<typename T::EnvironmentType &>(),
  //     std::declval<const typename T::KeyType &>(),
  //     std::declval<const typename T::KeyType &>());
};

template <typename CRTP, policy::objectives::isFiniteStateValueFunction VALUE_FUNCTION_T>
struct TDValueUpdaterBase {

  SETUP_TYPES_FROM_NESTED_ENVIRON(SINGLE_ARG(VALUE_FUNCTION_T::EnvironmentType));
  using KeyMaker = typename VALUE_FUNCTION_T::KeyMaker;

  void initialize(typename VALUE_FUNCTION_T::EnvironmentType &environment, VALUE_FUNCTION_T &valueFunction) {
    valueFunction.initialize(environment);
  }

  struct StatefulUpdateResult {
    bool isDone;
    ActionSpace action;
    TransitionType transition;
    PrecisionType reward;
  };

  void updateValue(
      VALUE_FUNCTION_T &valueFunction,
      policy::isFinitePolicyValueFunctionMixin auto &policy,
      policy::isFinitePolicyValueFunctionMixin auto &target_policy,
      typename VALUE_FUNCTION_T::EnvironmentType &environment,
      const typename VALUE_FUNCTION_T::KeyType &keyCurrent,
      const typename VALUE_FUNCTION_T::KeyType &keyNext,
      const typename VALUE_FUNCTION_T::PrecisionType &reward,
      const typename VALUE_FUNCTION_T::PrecisionType &discountRate) {

    valueFunction[keyCurrent].value =
        valueFunction.valueAt(keyCurrent) +
        temporal_differenc_error(
            valueFunction.valueAt(keyCurrent), valueFunction.valueAt(keyNext), reward, discountRate);
    valueFunction[keyCurrent].step++;
  }

  std::pair<bool, ActionSpace> update(
      VALUE_FUNCTION_T &valueFunction,
      policy::isFinitePolicyValueFunctionMixin auto &policy,
      policy::isFinitePolicyValueFunctionMixin auto &target_policy,
      EnvironmentType &environment,
      const ActionSpace &action,
      const PrecisionType &discountRate) {

    const auto [isDone, nextAction, transition, reward] =
        static_cast<CRTP *>(this)->step(valueFunction, policy, target_policy, environment, action, discountRate);

    // Update the value function.
    static_cast<CRTP *>(this)->updateValue(
        valueFunction,
        policy,
        target_policy,
        environment,
        KeyMaker::make(environment, transition.state, transition.action),
        KeyMaker::make(environment, transition.nextState, nextAction),
        reward,
        discountRate);

    return {transition.isDone(), nextAction};
  }
};

} // namespace temporal_difference