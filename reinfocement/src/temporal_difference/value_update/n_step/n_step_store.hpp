#pragma once

#include "temporal_difference/value_update/n_step/n_step_updater.hpp"

namespace temporal_difference {

template <typename CRTP, typename StatefulUpdateT>
struct ExpandedStateBuilder {
  struct ExpandedStatefulUpdateResult : public StatefulUpdateResult, public StorageInterface::AdditionalData {};
};

/// @brief  Default Interface is always to not add any additional storage.
/// @tparam CRTP
template <typename CRTP>
struct DefaultCRTPInterface {

  struct AdditionalData {};
  auto store(
      const VALUE_FUNCTION_T &valueFunction,
      policy::isFinitePolicyValueFunctionMixin auto &policy,
      policy::isFinitePolicyValueFunctionMixin auto &target_policy,
      const std::size_t &T,
      const std::size_t &n,
      const std::size_t &t,
      const PrecisionType &discountRate,
      const boost::circular_buffer<typename CRTP::ExpandedStatefulUpdateResult> &expandedTransitions)
      -> AdditionalData {
    return {};
  }
};

/// @brief Assume we want to progressively store the importance ratio for each state-action pairing up to the current
/// time - 1
/// @tparam CRTP
template <typename CRTP>
struct ProgressiveImportanceSamplingCRTPInterface {

  struct AdditionalData {
    const PrecisionType importanceRatio;
  };

  auto store(
      const VALUE_FUNCTION_T &valueFunction,
      policy::isFinitePolicyValueFunctionMixin auto &policy,
      policy::isFinitePolicyValueFunctionMixin auto &target_policy,
      const std::size_t &T,
      const std::size_t &n,
      const std::size_t &t,
      const PrecisionType &discountRate,
      const boost::circular_buffer<typename CRTP::ExpandedStatefulUpdateResult> &expandedTransitions)
      -> AdditionalData {

    // return additional data
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

    return {importanceRatio};
  }
};

/// @brief Store each partial importance sample such that we can create the total importance ratio at any time
/// @tparam CRTP
template <typename CRTP>
struct MarginalImportanceSamplingCRTPInterface {

  struct AdditionalData {
    const PrecisionType marginalImportanceRatio;
  };

  auto store(
      const VALUE_FUNCTION_T &valueFunction,
      policy::isFinitePolicyValueFunctionMixin auto &policy,
      policy::isFinitePolicyValueFunctionMixin auto &target_policy,
      const std::size_t &T,
      const std::size_t &n,
      const std::size_t &t,
      const PrecisionType &discountRate,
      const boost::circular_buffer<typename CRTP::ExpandedStatefulUpdateResult> &expandedTransitions)
      -> AdditionalData {

    // return additional data
    if (&policy == &target_policy)
      return 1.0F;

    const auto &data = expandedTransitions.at(t + 1);
    const auto marginalIImportanceRatio = target_policy(data.transition.state, data.transition.action) /
                                          policy(data.transition.state, data.transition.action);

    return {marginalIImportanceRatio};
  }
};

template <typename CRTP>
struct ReturnCalculationCRTPInterface {};

using NStepSarsaOffPolicy =
    NStepUpdater<void, ProgressiveImportanceSamplingCRTPInterface, ReturnCalculationCRTPInterface>

} // namespace temporal_difference