#pragma once
#include <concepts>

#include <boost/circular_buffer.hpp>

#include "reinforce/monte_carlo/episode.hpp"
#include "reinforce/temporal_difference/value_update/value_update.hpp"

namespace temporal_difference {

template <typename StatefulUpdateT, typename AdditionalData>
struct ExpandedStateBuilder {
  struct ExpandedStatefulUpdateResult : public StatefulUpdateT, public AdditionalData {
    ExpandedStatefulUpdateResult(const StatefulUpdateT &statefulUpdate, const AdditionalData &additionalData)
        : StatefulUpdateT{statefulUpdate}, AdditionalData{additionalData} {}
  };
};

template <typename Tp>
concept isNStepTemporalDifferenceStorageInterface = requires(
    Tp obj,
    const typename Tp::ValueFunctionType &valueFunction,
    policy::FinitePolicyValueFunctionMixin<typename Tp::ValueFunctionType> &policy,
    policy::FinitePolicyValueFunctionMixin<typename Tp::ValueFunctionType> &target_policy,
    typename Tp::EnvironmentType &environment,
    const typename Tp::ActionSpace &action,
    const typename Tp::PrecisionType &discountRate,
    const std::size_t &n,
    const std::size_t &t,
    const typename Tp::StatefulUpdateResult &statefulUpdateResult) {
  {
    obj.store(valueFunction, policy, target_policy, environment, discountRate, n, t, statefulUpdateResult)
  } -> std::same_as<typename Tp::AdditionalData>;
};

template <typename CRTP>
struct DefaultStorageInterface {

  SETUP_TYPES_W_VALUE_FUNCTION(CRTP::ValueFunctionType);

  struct AdditionalData {};
  auto store(
      const ValueFunctionType &valueFunction,
      policy::isFinitePolicyValueFunctionMixin auto &policy,
      policy::isFinitePolicyValueFunctionMixin auto &target_policy,
      EnvironmentType &environment,
      const PrecisionType &discountRate,
      const std::size_t &n,
      const std::size_t &t,
      const typename CRTP::StatefulUpdateResult &statefulUpdateResult) -> AdditionalData {
    return {};
  }
};

template <typename T>
concept isNStepTemporalDifferenceReturnCalculator = requires(
    T t,
    typename T::ValueFunctionType &valueFunction,
    policy::FinitePolicyValueFunctionMixin<typename T::ValueFunctionType> &policy,
    policy::FinitePolicyValueFunctionMixin<typename T::ValueFunctionType> &target_policy,
    typename T::EnvironmentType &environment,
    const typename T::PrecisionType &discountRate,
    typename boost::circular_buffer<typename T::ExpandedStatefulUpdateResult>::iterator start,
    typename boost::circular_buffer<typename T::ExpandedStatefulUpdateResult>::iterator end) {
  typename T::ReturnMetrics;

  {
    t.calculateReturn(valueFunction, policy, target_policy, environment, discountRate, start, end)
  } -> std::same_as<typename T::ReturnMetrics>;
};

template <typename T>
concept isNStepTemporalDifferenceValueUpdater = requires(
    T t,
    typename T::ValueFunctionType &valueFunction,
    policy::FinitePolicyValueFunctionMixin<typename T::ValueFunctionType> &policy,
    policy::FinitePolicyValueFunctionMixin<typename T::ValueFunctionType> &target_policy,
    typename T::EnvironmentType &environment,
    const typename T::PrecisionType &discountRate,
    const typename T::KeyType &keyCurrent,
    const typename T::ReturnMetrics &G) {
  {
    t.updateValue(valueFunction, policy, target_policy, environment, discountRate, keyCurrent, G)
  } -> std::same_as<void>;
};

template <typename CRTP>
struct DefaultNStepValueUpdater {

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
        alpha * temporal_differenc_error(valueFunction.valueAt(keyCurrent), 0.0F, G.ret, 0.0F);
  }
};

template <policy::objectives::isFiniteStateValueFunction V, template <typename CRTP> class StorageInterface>
struct NStepTemporalDifferenceValueUpdaterBase : TemporalDifferenceValueUpdaterBase<V> {

  SETUP_TYPES_W_VALUE_FUNCTION(V);

  using StatefulUpdateResult = typename TemporalDifferenceValueUpdaterBase<V>::StatefulUpdateResult;
  using ExpandedStatefulUpdateResult = typename ExpandedStateBuilder<
      typename TemporalDifferenceValueUpdaterBase<V>::StatefulUpdateResult,
      typename StorageInterface<TemporalDifferenceValueUpdaterBase<V>>::AdditionalData>::ExpandedStatefulUpdateResult;
};

template <
    policy::objectives::isFiniteStateValueFunction VALUE_FUNCTION_T,
    template <typename CRTP>
    typename StepInterface,
    template <typename CRTP>
    class StorageInterface,
    template <typename CRTP>
    class ReturnCalculationInterface,
    template <typename CRTP>
    class ValueUpdateInterface>
struct NStepUpdater
    : StepInterface<TemporalDifferenceValueUpdaterBase<VALUE_FUNCTION_T>>,
      StorageInterface<TemporalDifferenceValueUpdaterBase<VALUE_FUNCTION_T>>,
      ReturnCalculationInterface<NStepTemporalDifferenceValueUpdaterBase<VALUE_FUNCTION_T, StorageInterface>>,
      ValueUpdateInterface<
          ReturnCalculationInterface<NStepTemporalDifferenceValueUpdaterBase<VALUE_FUNCTION_T, StorageInterface>>> {

  static_assert(isTDUpdaterStep<StepInterface<TemporalDifferenceValueUpdaterBase<VALUE_FUNCTION_T>>>);
  static_assert(
      isNStepTemporalDifferenceReturnCalculator<
          ReturnCalculationInterface<NStepTemporalDifferenceValueUpdaterBase<VALUE_FUNCTION_T, StorageInterface>>>);

  SETUP_TYPES_W_VALUE_FUNCTION(VALUE_FUNCTION_T);

  using This =
      NStepUpdater<VALUE_FUNCTION_T, StepInterface, StorageInterface, ReturnCalculationInterface, ValueUpdateInterface>;
  using ExpandedStatefulUpdateResult =
      typename NStepTemporalDifferenceValueUpdaterBase<VALUE_FUNCTION_T, StorageInterface>::
          ExpandedStatefulUpdateResult;
  using BufferType = boost::circular_buffer<ExpandedStatefulUpdateResult>;
  using ReturnMetrics = typename ReturnCalculationInterface<
      NStepTemporalDifferenceValueUpdaterBase<VALUE_FUNCTION_T, StorageInterface>>::ReturnMetrics;

  std::size_t n;

  NStepUpdater(const std::size_t &n) : n{n} {}

  std::size_t moveByOffset(const std::size_t &n, const std::size_t t, const std::size_t &x) {
    // move t into position n
    return x - t + n;
  }

  int getTau(const std::size_t &n, const std::size_t &t) const { return t - n + 1; }

  bool hasReachedTerminalObservation(const boost::circular_buffer<ExpandedStatefulUpdateResult> &expandedTransitions) {
    if (expandedTransitions.size() > 0) {
      if (expandedTransitions.back().isDone) {
        // The terminal state had already been reached prior.
        return true;
      }
    }
    return false;
  }

  bool hasReachedTerminalValueUpdate(
      const boost::circular_buffer<ExpandedStatefulUpdateResult> &expandedTransitions,
      const std::size_t &n,
      const std::size_t &t) {
    if (expandedTransitions.size() > 0 and t == expandedTransitions.size() - 1 and expandedTransitions.back().isDone) {
      return true;
    }
    return false;
  }

  void observe(
      ValueFunctionType &valueFunction,
      policy::isFinitePolicyValueFunctionMixin auto &policy,
      policy::isFinitePolicyValueFunctionMixin auto &target_policy,
      EnvironmentType &environment,
      const ActionSpace &action,
      const PrecisionType &discountRate,
      const std::size_t &n,
      const std::size_t &t,
      boost::circular_buffer<ExpandedStatefulUpdateResult> &expandedTransitions) {

    // T is supposed to be initialised at infinity - we dont know when the episode ends.
    // At some stage a terminal state is reached.
    // For calculating prior returns we basically discount up to T (and no firther since it is terminal).
    // At the terminal state the return is just the reward.
    // Our data structure doesnt really need to keep track of T. It just needs to know if the state is terminal.
    // When the state is terminal we know it is T... and we can stop taking actions.
    // Simply put this function only should be called when expandedTransitions final element is NON-Terminal.

    //  take action, get the next action ready for storage into the transition buffer
    const auto s = this->step(valueFunction, policy, target_policy, environment, action, discountRate);

    // We stop taking actions if we reach a terminal state
    if (s.isDone) {
      // Return some signal here to indicate that we have reached a terminal state.
    } else {
      // add transition to buffer
      const auto additional = this->store(valueFunction, policy, target_policy, environment, discountRate, n, t, s);
      expandedTransitions.push_back(ExpandedStatefulUpdateResult{s, additional});
    }
    // std::cout << "observe: t:" << t << " size:" << expandedTransitions.size() << std::endl;
  }

  using ValueUpdateInterface<ReturnCalculationInterface<
      NStepTemporalDifferenceValueUpdaterBase<VALUE_FUNCTION_T, StorageInterface>>>::updateValue;

  void updateValue(
      VALUE_FUNCTION_T &valueFunction,
      policy::isFinitePolicyValueFunctionMixin auto &policy,
      policy::isFinitePolicyValueFunctionMixin auto &target_policy,
      EnvironmentType &environment,
      const ActionSpace &action,
      const PrecisionType &discountRate,
      const std::size_t &n,
      const std::size_t &t,
      const int &tau,
      boost::circular_buffer<ExpandedStatefulUpdateResult> &expandedTransitions) {

    // This update looks at the next n steps and calculates the return discounted back to this time period tau
    // However it is possible that the episode ends before n steps have been taken. In this case we discount back to
    // the end of the episode (T).
    //
    // When the terminal state is reached we no longer need to observe additional transitions - BUT we do need to
    // continue to incremenent tau and update the value function for it up to the terminal state.

    // if tau is negative then we have not yet reached the point where we can calculate the return (as we havent yet
    // taken n steps). In this case we do nothing.
    if (tau >= 0) {

      const auto tauBufferOffset = moveByOffset(n, t, tau);

      const auto discountedReturnG = [&]() -> ReturnMetrics {
        if (expandedTransitions.size() == 0)
          return {};

        auto bufferLowerBound = (expandedTransitions.begin() + tauBufferOffset + 1);
        auto bufferUpperBound = expandedTransitions.end();
        const auto discountedReturnG = this->calculateReturn(
            valueFunction, policy, target_policy, environment, discountRate, bufferLowerBound, bufferUpperBound);

        return discountedReturnG;
      }();

      // update
      const auto &atStatefulUpdate = expandedTransitions.at(tauBufferOffset);
      const auto keyTau =
          KeyMaker::make(environment, atStatefulUpdate.transition.state, atStatefulUpdate.transition.action);
      updateValue(valueFunction, policy, target_policy, environment, discountRate, keyTau, discountedReturnG);
      // std::cout << "updating tau: " << tau << " with return: " << discountedReturnG << "\n";
    }
  }

  std::pair<bool, ActionSpace> update(
      VALUE_FUNCTION_T &valueFunction,
      policy::isFinitePolicyValueFunctionMixin auto &policy,
      policy::isFinitePolicyValueFunctionMixin auto &target_policy,
      EnvironmentType &environment,
      const ActionSpace &action,
      const PrecisionType &discountRate,
      const std::size_t &n,
      const int &t,
      boost::circular_buffer<ExpandedStatefulUpdateResult> &expandedTransitions) {

    if (not hasReachedTerminalObservation(expandedTransitions)) {
      observe(valueFunction, policy, target_policy, environment, action, discountRate, n, t, expandedTransitions);
    }

    const int tau = getTau(n, t);
    updateValue(
        valueFunction, policy, target_policy, environment, action, discountRate, n, t, tau, expandedTransitions);

    return {expandedTransitions.back().isDone, expandedTransitions.back().action};
  };

  void update(
      VALUE_FUNCTION_T &valueFunction,
      policy::isFinitePolicyValueFunctionMixin auto &policy,
      policy::isFinitePolicyValueFunctionMixin auto &target_policy,
      EnvironmentType &environment,
      const PrecisionType &discountRate,
      const std::size_t max_steps) {

    environment.reset();

    bool isTerminal = false;
    ActionSpace action = ActionSpace{};
    boost::circular_buffer<ExpandedStatefulUpdateResult> expandedTransitions{n};
    std::size_t t = 0;

    while (true) {
      std::tie(isTerminal, action) =
          update(valueFunction, policy, target_policy, environment, action, discountRate, n, t, expandedTransitions);
      t++;
      // Send a termination signal when tau = T - 1 (i.e. the last step before the terminal state)
      if (hasReachedTerminalValueUpdate(expandedTransitions, n, t) or t == max_steps) {
        // Send a termination signal here.
        break;
      }
    }
  }
};

} // namespace temporal_difference
