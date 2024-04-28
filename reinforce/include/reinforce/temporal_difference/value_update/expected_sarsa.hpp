#pragma once
#include <utility>

#include "reinforce/policy/objectives/finite_value_function.hpp"
#include "reinforce/temporal_difference/value_update/q_learning.hpp"
#include "reinforce/temporal_difference/value_update/value_update.hpp"

namespace temporal_difference {

template <typename CRTP>
struct ExpectedSarsaValueUpdateMixin {

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

template <policy::objectives::isFiniteStateValueFunction V>
requires policy::objectives::isStateActionKeymaker<typename V::KeyMaker>
using ExpectedSARSAUpdater = TemporalDifferenceValueUpdater<V, QLearningStepMixin, ExpectedSarsaValueUpdateMixin>;

} // namespace temporal_difference
