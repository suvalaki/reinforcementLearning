#pragma once

#include <boost/circular_buffer.hpp>

#include "monte_carlo/episode.hpp"
#include "temporal_difference/value_update/value_update.hpp"

namespace temporal_difference {

template <typename T>
concept isTDNStepValueUpdateBaseInterface = requires(T t) {
  typename T::EnvironmentType;
  typename T::ActionSpace;
  typename T::TransitionType;
  typename T::PrecisionType;
  typename T::EpisodeType;

  t.calculateReturn(
       // const VALUE_FUNCTION_T &valueFunction,
       std::declval<typename T::ValueFunctionType &&>(),
       // const POLICY_T0 &policy,
       std::declval<policy::FiniteGreedyPolicy<typename T::KeyMaker, typename T::ValueType> &>() policy,
       // const POLICY_T1 &target_policy,
       std::declval<policy::FiniteGreedyPolicy<typename T::KeyMaker, typename T::ValueType> &>() target_policy,
       // const std::size_t &t,
       std::declval<std::size_t &>(),
       // const std::size_t &n,
       std::declval<std::size_t &>(),
       // const PrecisionType &discountRate,
       std::declval<PrecisionType &>(),
       // const boost::circular_buffer<StatefulUpdateResult> &expandedTransitions
       std::declval<boost::circular_buffer<StatefulUpdateResult> &>())
      ->typename T::PrecisionType;
};

template <
    policy::objectives::isFiniteStateValueFunction VALUE_FUNCTION_T,
    typename StepInterface,
    template <typename CRTP>
    class StorageInterfaceT,
    template <typename CRTP>
    class ReturnCalculationInterfaceT>
struct NStepUpdater {

  using This = NStepUpdater<StorageInterfaceT, ReturnCalculationInterfaceT>;
  using PrecisionType = typename StepInterface::PrecisionType;
  using EnvironmentType = typename StepInterface::EnvironmentType;
  using StatefulUpdateResult = typename StepInterface::StatefulUpdateResult;

  using StorageInterface = StorageInterfaceT<This>;
  using ReturnCalculationInterface = ReturnCalculationInterfaceT<This>;

  struct ExpandedStatefulUpdateResult : public StatefulUpdateResult, public StorageInterface::AdditionalData {};

  void updateValue(
      const VALUE_FUNCTION_T &valueFunction,
      policy::isFinitePolicyValueFunctionMixin auto &policy,
      policy::isFinitePolicyValueFunctionMixin auto &target_policy,
      const std::size_t &T,
      const std::size_t &n,
      const std::size_t &t,
      const std::size_t &tau,
      const PrecisionType &discountRate,
      const boost::circular_buffer<StatefulUpdateResult> &expandedTransitions) {

    auto alpha = 1.0F;
    auto &[state, action, nextState] = expandedTransitions[tau];
    auto keyCurrent = KeyMaker::makeKey(environment, state, action);

    valueFunction[keyCurrent].value =
        valueFunction.valueAt(keyCurrent) +
        alpha * temporal_differenc_error(valueFunction.valueAt(keyCurrent), 0.0F, G, 0.0F);
  }

  void update(
      const VALUE_FUNCTION_T &valueFunction,
      policy::isFinitePolicyValueFunctionMixin auto &policy,
      policy::isFinitePolicyValueFunctionMixin auto &target_policy,
      const std::size_t &T,
      const std::size_t &n,
      const std::size_t &t,
      const PrecisionType &discountRate,
      const boost::circular_buffer<StatefulUpdateResult> &expandedTransitions) {

    if (t < T) {
      // take action
      const auto s = step(valueFunction, policy, target_policy, environment, action, discountRate);

      // We stop taking actions if we reach a terminal state
      if (s.isDone) {
        T = t + 1;
      } else {
        // add transition to buffer
        const auto additional = store(valueFunction, policy, target_policy, environment, action, discountRate, T, t);
        expandedTransitions.push_back({s, additional});
      }
    }

    const auto tau = t - n + 1;
    if (tau >= 0) {
      // calculate return
      const auto discountedReturnG =
          calculateReturn(valueFunction, policy, target_policy, T, n, tau, discountRate, expandedTransitions);

      // update
      updateValue(valueFunction, policy, target_policy, T, n, t, tau, discountRate, expandedTransitions);
    }
  };
};

} // namespace temporal_difference
