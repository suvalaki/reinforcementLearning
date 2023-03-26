#pragma once
#include <algorithm>

#include "temporal_difference/value_update/n_step/n_step_updater.hpp"

namespace temporal_difference {

template <policy::objectives::isFiniteStateValueFunction VALUE_FUNCTION_T>
requires policy::objectives::isStateActionKeymaker<typename VALUE_FUNCTION_T::KeyMaker>
struct NStepSarsa : public TDNStepValueUpdateBaseInterface<NStepSarsa> {

  PrecisionType calculateImportanceRatio(
      const VALUE_FUNCTION_T &valueFunction,
      policy::isFinitePolicyValueFunctionMixin auto &policy,
      policy::isFinitePolicyValueFunctionMixin auto &target_policy,
      const std::size_t &T,
      const std::size_t &n,
      const std::size_t &tau,
      const PrecisionType &discountRate,
      const boost::circular_buffer<StatefulUpdateResult> &expandedTransitions) {

    if (&policy == &target_policy)
      return 1.0F;

    const auto maximportancesamplingtimestep = std::min(tau + n - 1, T - 1);
    const auto importanceRatio = std::accumulate(
        expandedTransitions.begin() + tau + 1,
        expandedTransitions.begin() + maxImportanceSamplingTimeStep,
        1.0F,
        [&policy, &target_policy](const auto &acc, const auto &e) {
          return acc * target_policy(e.state, e.action) / policy(e.state, e.action);
        });
  }

  PrecisionType calculateReturn(
      const VALUE_FUNCTION_T &valueFunction,
      policy::isFinitePolicyValueFunctionMixin auto &policy,
      policy::isFinitePolicyValueFunctionMixin auto &target_policy,
      const std::size_t &T,
      const std::size_t &n,
      const std::size_t &tau,
      const PrecisionType &discountRate,
      const boost::circular_buffer<StatefulUpdateResult> &expandedTransitions) {

    const auto maxTimeStep = std::min(tau + n, T);
    const auto G = std::accumulate(
        expandedTransitions.begin() + tau + 1,
        expandedTransitions.begin() + maxTimeStep,
        0.0F,
        [this, &valueFunction, &discountRate, &tau](const auto &acc, const auto &e) {
          return acc + std::pow(discountRate, e.timeStep - tau) * e.reward;
        });

    if (tau + n < T) {
      // const auto &lastTransition = expandedTransitions.back();
      const auto &lastTransition = expandedTransitions.at(tau + n);
      G += std::pow(discountRate, n) * valueFunction(lastTransition.state, lastTransition.action);
    }

    return G;
  }
};

} // namespace temporal_difference
