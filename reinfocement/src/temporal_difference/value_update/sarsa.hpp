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

  template <policy::isFinitePolicyValueFunctionMixin POLICY_T0, policy::isFinitePolicyValueFunctionMixin POLICY_T1>
  requires std::is_same_v<typename VALUE_FUNCTION_T::KeyType, typename POLICY_T0::KeyType> std::pair<bool, ActionSpace>
  update(VALUE_FUNCTION_T &valueFunction,
         POLICY_T0 &policy,
         POLICY_T1 &target_policy,
         EnvironmentType &environment,
         const ActionSpace &action,
         const PrecisionType &discountRate) {

    // Take Action - By stepping the environment automatically updates the state.
    const auto transition = environment.step(action);
    environment.update(transition);
    const auto reward = RewardType::reward(transition);

    // Sample the next action from the behavior policy.
    const auto nextAction = policy(environment, environment.state);

    // Update the value function.
    this->updateValue(valueFunction,
                      environment,
                      KeyMaker::make(environment, transition.state, transition.action),
                      KeyMaker::make(environment, transition.nextState, nextAction),
                      reward,
                      discountRate);

    return {transition.isDone(), nextAction};
  }
};

} // namespace temporal_difference
