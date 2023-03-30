#pragma once
#include <algorithm>
#include <cmath>

#include "temporal_difference/value_update/n_step/n_step_updater.hpp"
#include "temporal_difference/value_update/sarsa.hpp"

namespace temporal_difference {

template <typename CRTP>
struct OnPolicySarsaReturn {

  SETUP_TYPES_W_VALUE_FUNCTION(CRTP::ValueFunctionType);
  using ExpandedStatefulUpdateResult = typename CRTP::ExpandedStatefulUpdateResult;

  struct ReturnMetrics {
    const PrecisionType ret = 0;
  };

  ReturnMetrics calculateReturn(
      const ValueFunctionType &valueFunction,
      policy::isFinitePolicyValueFunctionMixin auto &policy,
      policy::isFinitePolicyValueFunctionMixin auto &target_policy,
      EnvironmentType &environment,
      const PrecisionType &discountRate,
      typename boost::circular_buffer<ExpandedStatefulUpdateResult>::iterator start,
      typename boost::circular_buffer<ExpandedStatefulUpdateResult>::iterator end) {

    auto G = std::accumulate(
        start, end, 0.0F, [discountRate, &valueFunction, &environment](const auto &acc, const auto &itr) {
          return acc + std::pow(discountRate, 1) * itr.reward;
        });

    const auto last = end - 1;
    if (last->isDone)
      return {
          G + discountRate *
                  valueFunction(KeyMaker::make(environment, last->transition.state, last->transition.action)).value};

    return {G};
  }
};

template <policy::objectives::isFiniteStateValueFunction V>
requires policy::objectives::isStateActionKeymaker<typename V::KeyMaker>
using NStepSarsaUpdater =
    NStepUpdater<V, SarsaStepMixin, DefaultStorageInterface, OnPolicySarsaReturn, DefaultNStepValueUpdater>;

// /// @brief Assume we want to progressively store the importance ratio for each state-action pairing up to the current
// /// time - 1
// /// @tparam CRTP
// template <typename CRTP>
// struct ProgressiveImportanceSamplingStorageInterface {

//   struct AdditionalData {
//     const PrecisionType importanceRatio;
//   };

//   auto store(
//       const VALUE_FUNCTION_T &valueFunction,
//       policy::isFinitePolicyValueFunctionMixin auto &policy,
//       policy::isFinitePolicyValueFunctionMixin auto &target_policy,
//       const std::size_t &T,
//       const std::size_t &n,
//       const std::size_t &t,
//       const PrecisionType &discountRate,
//       const boost::circular_buffer<typename CRTP::ExpandedStatefulUpdateResult> &expandedTransitions)
//       -> AdditionalData {

//     // return additional data
//     if (&policy == &target_policy)
//       return 1.0F;

//     const auto maximportancesamplingtimestep = std::min(tau + n - 1, T - 1);
//     const auto importanceRatio = std::accumulate(
//         expandedTransitions.begin() + tau + 1,
//         expandedTransitions.begin() + maxImportanceSamplingTimeStep,
//         1.0F,
//         [&policy, &target_policy](const auto &acc, const auto &e) {
//           return acc * target_policy(e.state, e.action) / policy(e.state, e.action);
//         });

//     return {importanceRatio};
//   }
// };

// template <typename CRTP>
// struct OffPolicySarsaReturnCalculator {

//   PrecisionType calculateReturn(
//       const VALUE_FUNCTION_T &valueFunction,
//       policy::isFinitePolicyValueFunctionMixin auto &policy,
//       policy::isFinitePolicyValueFunctionMixin auto &target_policy,
//       const std::size_t &T,
//       const std::size_t &n,
//       const std::size_t &tau,
//       const PrecisionType &discountRate,
//       const boost::circular_buffer<typename CRTP::ExpandedStatefulUpdateResult> &expandedTransitions) {

//     const auto maximportancesamplingtimestep = std::min(tau + n - 1, T - 1);
//     const auto importanceRatio = std::accumulate(
//         expandedTransitions.begin() + tau + 1,
//         expandedTransitions.begin() + maxImportanceSamplingTimeStep,
//         1.0F,
//         [&policy, &target_policy](const auto &acc, const auto &e) {
//           return acc * target_policy(e.state, e.action) / policy(e.state, e.action);
//         });

//     const auto maxtimestep = std::min(tau + n, T);
//     const auto return = std::accumulate(
//         expandedTransitions.begin() + tau,
//         expandedTransitions.begin() + maxtimestep,
//         0.0F,
//         [discountRate](const auto &acc, const auto &e) { return acc + e.reward + discountRate * e.nextValue; });

//     return importanceRatio * return;
//   }
// };

} // namespace temporal_difference
