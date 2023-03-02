#pragma once
#include <utility>

#include "policy/objectives/finite_value_function.hpp"
#include "temporal_difference/value_update/q_learning.hpp"
#include "temporal_difference/value_update/td0_updater.hpp"

namespace temporal_difference {

template <policy::objectives::isFiniteStateValueFunction VALUE_FUNCTION_T>
requires policy::objectives::isStateActionKeymaker<typename VALUE_FUNCTION_T::KeyMaker>
struct ExpectedSARSAUpdater : QLearningUpdater<VALUE_FUNCTION_T> {

  // Uses the same update as Q-Learning But with a different Value update method

  // This is SARSA when valueFunction == policy

  SETUP_TYPES_FROM_NESTED_ENVIRON(SINGLE_ARG(VALUE_FUNCTION_T::EnvironmentType));
  using KeyMaker = typename VALUE_FUNCTION_T::KeyMaker;

  using QLearningUpdater<VALUE_FUNCTION_T>::q_learning_step;

  void updateValue(VALUE_FUNCTION_T &valueFunction,
                   policy::isFinitePolicyValueFunctionMixin auto &policy,
                   typename VALUE_FUNCTION_T::EnvironmentType &environment,
                   const VALUE_FUNCTION_T::KeyType &keyCurrent,
                   const typename VALUE_FUNCTION_T::PrecisionType &reward,
                   const typename VALUE_FUNCTION_T::PrecisionType &discountRate) {

    // get the max value from the next state.
    // This is the difference between SARSA and Q-Learning.
    const auto reachableActions = environment.getReachableActions(environment.state);
    const auto expectedNextValue = std::accumulate(
        reachableActions.begin(),
        reachableActions.end(),
        std::numeric_limits<PrecisionType>::lowest(),
        [&](const auto &a, const auto &action) {
          const auto val =
              policy.getProbability(environment, KeyMaker::get_state_from_key(environment, keyCurrent), action) *
              valueFunction.valueAt(KeyMaker::make(environment, environment.state, action));
          return a + val;
        });

    valueFunction[keyCurrent].value =
        valueFunction.valueAt(keyCurrent) +
        temporal_differenc_error(valueFunction.valueAt(keyCurrent), expectedNextValue, reward, discountRate);
    valueFunction[keyCurrent].step++;
  }

  template <policy::isFinitePolicyValueFunctionMixin POLICY_T0, policy::isFinitePolicyValueFunctionMixin POLICY_T1>
  requires std::is_same_v<typename VALUE_FUNCTION_T::KeyType, typename POLICY_T0::KeyType> std::pair<bool, ActionSpace>
  update(VALUE_FUNCTION_T &valueFunction,
         POLICY_T0 &policy,
         POLICY_T1 &target_policy,
         EnvironmentType &environment,
         const ActionSpace &action,
         const PrecisionType &discountRate) {

    const auto [isDone, nextAction, transition, reward] =
        q_learning_step(valueFunction, policy, environment, action, discountRate);

    // Update the value function.
    updateValue(valueFunction,
                policy,
                environment,
                KeyMaker::make(environment, transition.state, transition.action),
                reward,
                discountRate);

    return {transition.isDone(), nextAction};
  }
};

} // namespace temporal_difference
