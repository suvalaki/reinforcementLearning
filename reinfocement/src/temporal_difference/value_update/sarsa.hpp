#pragma once
#include <utility>

#include "policy/objectives/finite_value_function.hpp"
#include "temporal_difference/value_update/td0_updater.hpp"

namespace temporal_difference {

template <policy::objectives::isFiniteStateValueFunction VALUE_FUNCTION_T>
requires policy::objectives::isStateActionKeymaker<typename VALUE_FUNCTION_T::KeyMaker>
struct SARSAUpdater : TDValueUpdaterBase<SARSAUpdater<VALUE_FUNCTION_T>, VALUE_FUNCTION_T> {

  // This is SARSA when valueFunction == policy

  SETUP_TYPES_FROM_NESTED_ENVIRON(SINGLE_ARG(VALUE_FUNCTION_T::EnvironmentType));
  using KeyMaker = typename VALUE_FUNCTION_T::KeyMaker;
  using StatefulUpdateResult =
      typename TDValueUpdaterBase<SARSAUpdater<VALUE_FUNCTION_T>, VALUE_FUNCTION_T>::StatefulUpdateResult;

  StatefulUpdateResult step(
      VALUE_FUNCTION_T &valueFunction,
      policy::isFinitePolicyValueFunctionMixin auto &policy,
      policy::isFinitePolicyValueFunctionMixin auto &target_policy,
      EnvironmentType &environment,
      const ActionSpace &action,
      const PrecisionType &discountRate);
};

template <policy::objectives::isFiniteStateValueFunction VALUE_FUNCTION_T>
auto SARSAUpdater<VALUE_FUNCTION_T>::step(
    VALUE_FUNCTION_T &valueFunction,
    policy::isFinitePolicyValueFunctionMixin auto &policy,
    policy::isFinitePolicyValueFunctionMixin auto &target_policy,
    EnvironmentType &environment,
    const ActionSpace &action,
    const PrecisionType &discountRate) -> StatefulUpdateResult {

  // Take Action - By stepping the environment automatically updates the state.
  const auto transition = environment.step(action);
  environment.update(transition);
  const auto reward = RewardType::reward(transition);

  // Sample the next action from the behavior policy.
  const auto nextAction = policy(environment, environment.state);

  return {transition.isDone(), nextAction, transition, reward};
}

} // namespace temporal_difference
