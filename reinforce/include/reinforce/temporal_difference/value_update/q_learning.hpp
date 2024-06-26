#pragma once
#include <limits>
#include <numeric>
#include <tuple>
#include <utility>

#include "reinforce/policy/objectives/finite_value_function.hpp"
#include "reinforce/temporal_difference/value_update/value_update.hpp"

namespace temporal_difference {

template <typename CRTP>
struct QLearningStepMixin {

  SETUP_TYPES_W_VALUE_FUNCTION(CRTP::ValueFunctionType);

  auto step(
      typename CRTP::ValueFunctionType &valueFunction,
      policy::isFinitePolicyValueFunctionMixin auto &policy,
      policy::isFinitePolicyValueFunctionMixin auto &target_policy,
      EnvironmentType &environment,
      const ActionSpace &action,
      const PrecisionType &discountRate) -> typename CRTP::StatefulUpdateResult {

    // Take Action - By stepping the environment automatically updates the state.
    // Sample the next action from the behavior policy.
    const auto nextAction = policy(environment, environment.state);

    // Take Action - By stepping the environment automatically updates the state.
    const auto transition = environment.step(nextAction);
    environment.update(transition);
    const auto reward = RewardType::reward(transition);

    return {transition.isDone(), action, transition, reward};
  }
};

template <typename CRTP>
struct QLearningValueUpdateMixin {

  SETUP_TYPES_W_VALUE_FUNCTION(CRTP::ValueFunctionType);

  void updateValue(
      typename CRTP::ValueFunctionType &valueFunction,
      policy::isFinitePolicyValueFunctionMixin auto &policy,
      policy::isFinitePolicyValueFunctionMixin auto &target_policy,
      EnvironmentType &environment,
      const KeyType &keyCurrent,
      const KeyType &keyNext,
      const PrecisionType &reward,
      const PrecisionType &discountRate) {

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
};

template <policy::objectives::isFiniteStateValueFunction V>
requires policy::objectives::isStateActionKeymaker<typename V::KeyMaker>
using QLearningUpdater = TemporalDifferenceValueUpdater<V, QLearningStepMixin, QLearningValueUpdateMixin>;

} // namespace temporal_difference
