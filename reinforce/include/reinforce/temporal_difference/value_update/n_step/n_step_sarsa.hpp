#pragma once
#include <algorithm>
#include <cmath>

#include "reinforce/temporal_difference/value_update/n_step/n_step_updater.hpp"
#include "reinforce/temporal_difference/value_update/sarsa.hpp"

/**
 * n-step policy return can be written recursively:
 *
 *    G_t = R_{t+1} + \gamma * ( rho_{t+1} G_{t+1 : h} + V_{h-1} (S_{t+1}) - rho_{t+1} Q_{h-1} (S_{t+1}, A_{t+1})  )
 *
 */

namespace temporal_difference {

template <typename CRTP>
struct OnPolicySarsaReturn {

  SETUP_TYPES_W_VALUE_FUNCTION(CRTP::ValueFunctionType);
  using ExpandedStatefulUpdateResult = typename CRTP::ExpandedStatefulUpdateResult;

  struct ReturnMetrics {
    const PrecisionType ret = 0;
  };

  ReturnMetrics calculateReturn(
      ValueFunctionType &valueFunction,
      policy::isFinitePolicyValueFunctionMixin auto &policy,
      policy::isFinitePolicyValueFunctionMixin auto &target_policy,
      EnvironmentType &environment,
      const PrecisionType &discountRate,
      typename boost::circular_buffer<ExpandedStatefulUpdateResult>::iterator start,
      typename boost::circular_buffer<ExpandedStatefulUpdateResult>::iterator end) {

    auto G = std::accumulate(
        start,
        end,
        0.0F,
        [discountRate, &valueFunction, &environment, currentDiscountRate = 1.0F](
            const auto &acc, const auto &itr) mutable {
          currentDiscountRate *= discountRate;
          return acc + currentDiscountRate * itr.reward;
        });

    const auto last = end - 1;
    if (last->isDone)
      return {
          G + discountRate *
                  valueFunction.valueAt(KeyMaker::make(environment, last->transition.state, last->transition.action))};

    return {G};
  }
};

template <policy::objectives::isFiniteStateValueFunction V>
requires policy::objectives::isStateActionKeymaker<typename V::KeyMaker>
using NStepSarsaUpdater =
    NStepUpdater<V, SarsaStepMixin, DefaultStorageInterface, OnPolicySarsaReturn, DefaultNStepValueUpdater>;

/// @brief Assume we want to progressively store the importance ratio for each state-action pairing up to the current
/// time - 1
/// @tparam CRTP
template <typename CRTP>
struct ProgressiveImportanceSamplingStorageInterface {

  SETUP_TYPES_W_VALUE_FUNCTION(CRTP::ValueFunctionType);

  struct AdditionalData {
    PrecisionType importanceRatio;
  };

  auto store(
      const ValueFunctionType &valueFunction,
      policy::isFinitePolicyValueFunctionMixin auto &policy,
      policy::isFinitePolicyValueFunctionMixin auto &target_policy,
      EnvironmentType &environment,
      const PrecisionType &discountRate,
      const std::size_t &n,
      const std::size_t &t,
      const typename CRTP::StatefulUpdateResult &statefulUpdateResult) -> AdditionalData {

    // return additional data
    if (&policy == &target_policy)
      return {1.0F};

    const auto &e = statefulUpdateResult.transition;
    return {
        target_policy.getProbability(environment, e.state, e.action) /
        policy.getProbability(environment, e.state, e.action)};
  }
};

template <typename CRTP>
struct OffPolicySarsaReturnCalculator {

  SETUP_TYPES_W_VALUE_FUNCTION(CRTP::ValueFunctionType);
  using ExpandedStatefulUpdateResult = typename CRTP::ExpandedStatefulUpdateResult;
  struct ReturnMetrics {
    const PrecisionType ret = 0;
    const PrecisionType importanceRatio = 0;
  };

  ReturnMetrics calculateReturn(
      ValueFunctionType &valueFunction,
      policy::isFinitePolicyValueFunctionMixin auto &policy,
      policy::isFinitePolicyValueFunctionMixin auto &target_policy,
      EnvironmentType &environment,
      const PrecisionType &discountRate,
      typename boost::circular_buffer<ExpandedStatefulUpdateResult>::iterator start,
      typename boost::circular_buffer<ExpandedStatefulUpdateResult>::iterator end) {

    const auto G = std::accumulate(start, end, 0.0F, [&discountRate](const auto &acc, const auto &itr) {
      return acc + std::pow(discountRate, 1) * itr.reward;
    });

    const auto rho =
        start == end ? 1.0F : std::accumulate(start, (end - 1), 1.0F, [](const auto &acc, const auto &itr) {
          return acc * itr.importanceRatio;
        });

    if (const auto last = end - 1; last->isDone)
      return {
          .ret = G + discountRate * valueFunction.valueAt(
                                        KeyMaker::make(environment, last->transition.state, last->transition.action)),
          .importanceRatio = rho};

    return {.ret = G, .importanceRatio = rho};
  }
};

template <typename CRTP>
struct OffPolicySarsaNStepValueUpdater {

  SETUP_TYPES_W_VALUE_FUNCTION(CRTP::ValueFunctionType);
  using ReturnMetrics = typename CRTP::ReturnMetrics;

  void updateValue(
      ValueFunctionType &valueFunction,
      policy::isFinitePolicyValueFunctionMixin auto &policy,
      policy::isFinitePolicyValueFunctionMixin auto &target_policy,
      EnvironmentType &environment,
      const PrecisionType &discountRate,
      const KeyType &keyCurrent,
      const ReturnMetrics &G) {

    const auto alpha = 1.0F;
    valueFunction[keyCurrent].value =
        valueFunction.valueAt(keyCurrent) +
        alpha * G.importanceRatio * temporal_differenc_error(valueFunction.valueAt(keyCurrent), 0.0F, G.ret, 0.0F);
  }
};

template <policy::objectives::isFiniteStateValueFunction V>
requires policy::objectives::isStateActionKeymaker<typename V::KeyMaker>
using NStepSarsaOffPolicyUpdater = NStepUpdater<
    V,
    SarsaStepMixin,
    ProgressiveImportanceSamplingStorageInterface,
    OffPolicySarsaReturnCalculator,
    OffPolicySarsaNStepValueUpdater>;

} // namespace temporal_difference
