#pragma once

#include "reinforce/policy/distribution_policy.hpp"
#include "reinforce/policy/finite/value_policy.hpp"
#include "reinforce/policy/objectives/finite_value_function.hpp"
#include "reinforce/policy/value.hpp"
#include "reinforce/temporal_difference/value_update/value_update.hpp"

namespace temporal_difference {

template <
    policy::objectives::isFiniteStateValueFunction VALUE_FUNCTION_T,
    policy::isFinitePolicyValueFunctionMixin POLICY_T0,
    policy::isFinitePolicyValueFunctionMixin POLICY_T1,
    typename VALUE_UPDATER_T>
void one_step_valueEstimate_episode(
    VALUE_FUNCTION_T &valueFunction,
    typename VALUE_FUNCTION_T::EnvironmentType &environment,
    POLICY_T0 &policy,
    POLICY_T1 &target_policy,
    VALUE_UPDATER_T &valueUpdater,
    const typename VALUE_FUNCTION_T::PrecisionType &discountRate = 1.0F,
    const std::size_t &maxSteps = 10) {

  valueUpdater.update(valueFunction, policy, target_policy, environment, discountRate, maxSteps);
}

template <
    policy::objectives::isFiniteStateValueFunction VALUE_FUNCTION_T,
    policy::isFinitePolicyValueFunctionMixin POLICY_T0,
    policy::isFinitePolicyValueFunctionMixin POLICY_T1,
    isTDValueUpdater VALUE_UPDATER_T>
void one_step_valueEstimate(
    VALUE_FUNCTION_T &valueFunction,
    typename VALUE_FUNCTION_T::EnvironmentType &environment,
    POLICY_T0 &policy,
    POLICY_T1 &target_policy,
    VALUE_UPDATER_T &valueUpdater,
    const std::size_t &episodes,
    const std::size_t &maxSteps = 10) {

  for (std::size_t episode = 0; episode < episodes; ++episode) {
    one_step_valueEstimate_episode(valueFunction, environment, policy, target_policy, valueUpdater, maxSteps);
  }
}

} // namespace temporal_difference
