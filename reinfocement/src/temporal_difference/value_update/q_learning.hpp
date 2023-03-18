#pragma once
#include <limits>
#include <numeric>
#include <tuple>
#include <utility>

#include "policy/objectives/finite_value_function.hpp"
#include "temporal_difference/value_update/td0_updater.hpp"

namespace temporal_difference {

template <policy::objectives::isFiniteStateValueFunction VALUE_FUNCTION_T>
requires policy::objectives::isStateActionKeymaker<typename VALUE_FUNCTION_T::KeyMaker>
struct QLearningUpdater : TDValueUpdaterBase<QLearningUpdater<VALUE_FUNCTION_T>, VALUE_FUNCTION_T> {

  // This is SARSA when valueFunction == policy

  SETUP_TYPES_FROM_NESTED_ENVIRON(SINGLE_ARG(VALUE_FUNCTION_T::EnvironmentType));
  using KeyMaker = typename VALUE_FUNCTION_T::KeyMaker;
  using StatefulUpdateResult =
      typename TDValueUpdaterBase<QLearningUpdater<VALUE_FUNCTION_T>, VALUE_FUNCTION_T>::StatefulUpdateResult;

  StatefulUpdateResult step(
      VALUE_FUNCTION_T &valueFunction,
      policy::isFinitePolicyValueFunctionMixin auto &policy,
      policy::isFinitePolicyValueFunctionMixin auto &target_policy,
      EnvironmentType &environment,
      const ActionSpace &action,
      const PrecisionType &discountRate);

  void updateValue(
      VALUE_FUNCTION_T &valueFunction,
      policy::isFinitePolicyValueFunctionMixin auto &policy,
      policy::isFinitePolicyValueFunctionMixin auto &target_policy,
      typename VALUE_FUNCTION_T::EnvironmentType &environment,
      const VALUE_FUNCTION_T::KeyType &keyCurrent,
      const VALUE_FUNCTION_T::KeyType &keyNext,
      const typename VALUE_FUNCTION_T::PrecisionType &reward,
      const typename VALUE_FUNCTION_T::PrecisionType &discountRate);
};

template <policy::objectives::isFiniteStateValueFunction VALUE_FUNCTION_T>
auto QLearningUpdater<VALUE_FUNCTION_T>::step(
    VALUE_FUNCTION_T &valueFunction,
    policy::isFinitePolicyValueFunctionMixin auto &policy,
    policy::isFinitePolicyValueFunctionMixin auto &target_policy,
    EnvironmentType &environment,
    const ActionSpace &action,
    const PrecisionType &discountRate) -> StatefulUpdateResult {

  // Take Action - By stepping the environment automatically updates the state.
  // Sample the next action from the behavior policy.
  const auto nextAction = policy(environment, environment.state);

  // Take Action - By stepping the environment automatically updates the state.
  const auto transition = environment.step(nextAction);
  environment.update(transition);
  const auto reward = RewardType::reward(transition);

  return {transition.isDone(), action, transition, reward};
}

template <policy::objectives::isFiniteStateValueFunction VALUE_FUNCTION_T>
auto QLearningUpdater<VALUE_FUNCTION_T>::updateValue(
    VALUE_FUNCTION_T &valueFunction,
    policy::isFinitePolicyValueFunctionMixin auto &policy,
    policy::isFinitePolicyValueFunctionMixin auto &target_policy,
    typename VALUE_FUNCTION_T::EnvironmentType &environment,
    const VALUE_FUNCTION_T::KeyType &keyCurrent,
    const VALUE_FUNCTION_T::KeyType &keyNext,
    const typename VALUE_FUNCTION_T::PrecisionType &reward,
    const typename VALUE_FUNCTION_T::PrecisionType &discountRate) -> void {

  // get the max value from the next state.
  // This is the difference between SARSA and Q-Learning.
  const auto reachableActions = environment.getReachableActions(environment.state);
  const auto maxNextValue = std::accumulate(
      reachableActions.begin(),
      reachableActions.end(),
      std::numeric_limits<PrecisionType>::lowest(),
      [&](const auto &a, const auto &action) {
        const auto val = valueFunction.valueAt(KeyMaker::make(environment, environment.state, action));
        if (val > a)
          return val;
        return a;
      });

  valueFunction[keyCurrent].value =
      valueFunction.valueAt(keyCurrent) +
      temporal_differenc_error(valueFunction.valueAt(keyCurrent), maxNextValue, reward, discountRate);
  valueFunction[keyCurrent].step++;
}

} // namespace temporal_difference
