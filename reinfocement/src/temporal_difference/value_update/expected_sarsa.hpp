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

  using QLearningUpdater<VALUE_FUNCTION_T>::step;

  void updateValue(
      VALUE_FUNCTION_T &valueFunction,
      policy::isFinitePolicyValueFunctionMixin auto &policy,
      policy::isFinitePolicyValueFunctionMixin auto &target_policy,
      typename VALUE_FUNCTION_T::EnvironmentType &environment,
      const VALUE_FUNCTION_T::KeyType &keyCurrent,
      const VALUE_FUNCTION_T::KeyType &keyNext,
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
};

} // namespace temporal_difference
