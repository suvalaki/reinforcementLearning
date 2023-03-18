#pragma once

#include "policy/distribution_policy.hpp"
#include "policy/finite/value_policy.hpp"
#include "policy/objectives/finite_value_function.hpp"
#include "policy/value.hpp"

#include "temporal_difference/value_update/value_update.hpp"

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

  environment.reset();

  auto action = policy(environment, environment.state);
  auto isTerminal = false;
  auto step = 0;

  while (not isTerminal and step < maxSteps) {
    std::tie(isTerminal, action) =
        valueUpdater.update(valueFunction, policy, target_policy, environment, action, discountRate);
    step++;
  };
}

#if false
template <
    policy::objectives::isFiniteStateValueFunction VALUE_FUNCTION_T,
    policy::isFinitePolicyValueFunctionMixin POLICY_T0,
    policy::isFinitePolicyValueFunctionMixin POLICY_T1,
    typename VALUE_UPDATER_T>
void n_step_valueEstimate_episode(
    VALUE_FUNCTION_T &valueFunction,
    typename VALUE_FUNCTION_T::EnvironmentType &environment,
    POLICY_T0 &policy,
    POLICY_T1 &target_policy,
    VALUE_UPDATER_T &valueUpdater,
    const typename VALUE_FUNCTION_T::PrecisionType &discountRate = 1.0F,
    const std::size_t &n = 10 const typename VALUE_FUNCTION_T::PrecisionType &samplingDegree = 0.5F) {

  environment.reset();

  auto action = policy(environment, environment.state);
  auto isTerminal = false;
  auto T = 2 * n;
  auto step = 0;
  auto tau = 0;
  auto importanceWeight = 1.0F;

  while (not isTerminal and step < T) {

    auto R = 0;
    auto state = environment.state;
    if (step < T) {
      // take action and store next reward r_t+1
      std::tie(isTerminal, action) =
          valueUpdater.update(valueFunction, policy, target_policy, environment, action, discountRate);
      // Store R_t
      if (isTerminal) {
        T = step + 1;
      } else {
        action = policy(environment, environment.state);
        // select and store sigma (t + 1)
        // store importance sampling ratio as rho (t+ 1)
      }
    }

    tau = step - n + 1;
    if (tau >= 0) {
      auto G = 0;
      for (auto k = std::min(step + 1, T); k < tau + 1; k++) {
        if (k == T) {
          G += R;
        } else {
          //
          const auto availableActions = environment.availableActions(state);
          const auto V = std::accumulate(
              availableActions.begin(), availableActions.end(), 0.0F, [&](const auto &acc, const auto &action) {
                return acc + target_policy.getProbability(environment, state, action) *
                                 valueFunction(environment, state, action);
              });
          G = R +
              discountRate *
                  (samplingDegree * importanceWeight +
                   (1 - samplingDegree) * target_policy.getProbability(environment, environment.state, action)) *
                  (G - valueFunction(environment, state, action)) +
              discountRate * V;
        }
      }
    }

    step++;
  };
}
#endif

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
