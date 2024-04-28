#pragma once
#include <utility>

#include "reinforce/policy/objectives/finite_value_function.hpp"
#include "reinforce/temporal_difference/value_update/value_update.hpp"

namespace temporal_difference {

template <typename CRTP>
struct SarsaStepMixin {

  SETUP_TYPES_W_VALUE_FUNCTION(CRTP::ValueFunctionType);

  auto step(
      typename CRTP::ValueFunctionType &valueFunction,
      policy::isFinitePolicyValueFunctionMixin auto &policy,
      policy::isFinitePolicyValueFunctionMixin auto &target_policy,
      EnvironmentType &environment,
      const ActionSpace &action,
      const PrecisionType &discountRate) -> typename CRTP::StatefulUpdateResult {

    // Sample the next action from the behavior policy.
    const auto nextAction = policy(environment, environment.state);

    // Take Action - By stepping the environment automatically updates the state.
    const auto transition = environment.step(action);
    environment.update(transition);
    const auto reward = RewardType::reward(transition);

    return {transition.isDone(), nextAction, transition, reward};
  }
};

template <policy::objectives::isFiniteStateValueFunction V>
requires policy::objectives::isStateActionKeymaker<typename V::KeyMaker>
using SARSAUpdater = TemporalDifferenceValueUpdater<V, SarsaStepMixin, DefaultValueUpdater>;

} // namespace temporal_difference
