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

  template <policy::isFinitePolicyValueFunctionMixin POLICY_T>
  std::tuple<bool, ActionSpace, TransitionType, PrecisionType> q_learning_step(VALUE_FUNCTION_T &valueFunction,
                                                                               POLICY_T &policy,
                                                                               EnvironmentType &environment,
                                                                               const ActionSpace &action,
                                                                               const PrecisionType &discountRate) {

    // Take Action - By stepping the environment automatically updates the state.
    // Sample the next action from the behavior policy.
    const auto nextAction = policy(environment, environment.state);

    // Take Action - By stepping the environment automatically updates the state.
    const auto transition = environment.step(nextAction);
    environment.update(transition);
    const auto reward = RewardType::reward(transition);

    return {transition.isDone(), action, transition, reward};
  }

  void updateValue(VALUE_FUNCTION_T &valueFunction,
                   typename VALUE_FUNCTION_T::EnvironmentType &environment,
                   const VALUE_FUNCTION_T::KeyType &keyCurrent,
                   const typename VALUE_FUNCTION_T::PrecisionType &reward,
                   const typename VALUE_FUNCTION_T::PrecisionType &discountRate) {

    // get the max value from the next state.
    // This is the difference between SARSA and Q-Learning.
    const auto reachableActions = environment.getReachableActions(environment.state);
    const auto maxNextValue = std::accumulate(reachableActions.begin(),
                                              reachableActions.end(),
                                              std::numeric_limits<PrecisionType>::lowest(),
                                              [&](const auto &a, const auto &action) {
                                                const auto val = valueFunction.valueAt(
                                                    KeyMaker::make(environment, environment.state, action));
                                                if (val > a)
                                                  return val;
                                                return a;
                                              });

    valueFunction[keyCurrent].value =
        valueFunction.valueAt(keyCurrent) +
        temporal_differenc_error(valueFunction.valueAt(keyCurrent), maxNextValue, reward, discountRate);
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
                environment,
                KeyMaker::make(environment, transition.state, transition.action),
                reward,
                discountRate);

    return {transition.isDone(), nextAction};
  }
};

} // namespace temporal_difference