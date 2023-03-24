#pragma once

#include <unordered_map>
#include <vector>

#include "monte_carlo/episode.hpp"
#include "policy/distribution_policy.hpp"
#include "policy/finite/greedy_policy.hpp"
#include "policy/finite/value_policy.hpp"
#include "policy/objectives/finite_value_function.hpp"
#include "policy/value.hpp"

namespace temporal_difference {

auto temporal_differenc_error(
    const auto &valueAtStateT, const auto &valueAtStateT1, const auto &reward, const auto &discountRate) {
  return reward + discountRate * valueAtStateT1 - valueAtStateT;
}

template <typename T>
concept isTDUpdaterStep = requires(
    T t,
    typename T::ValueFunctionType &valueFunction,
    policy::FinitePolicyValueFunctionMixin<typename T::ValueFunctionType> &policy,
    policy::FinitePolicyValueFunctionMixin<typename T::ValueFunctionType> &target_policy,
    typename T::EnvironmentType &environment,
    const typename T::ActionSpace &action,
    const typename T::PrecisionType &discountRate) {

  t.step(valueFunction, policy, target_policy, environment, action, discountRate);
};

template <typename T>
concept isTDUpdaterValueUpdate = requires(
    T t,
    typename T::ValueFunctionType &valueFunction,
    policy::FinitePolicyValueFunctionMixin<typename T::ValueFunctionType> &policy,
    policy::FinitePolicyValueFunctionMixin<typename T::ValueFunctionType> &target_policy,
    typename T::EnvironmentType &environment,
    const typename T::KeyType &keyCurrent,
    const typename T::KeyType &keyNext,
    const typename T::PrecisionType &reward,
    const typename T::PrecisionType &discountRate) {

  t.updateValue(valueFunction, policy, target_policy, environment, keyCurrent, keyNext, reward, discountRate);
};

template <typename T>
concept isTDUpdaterUpdate = requires(
    T t,
    typename T::ValueFunctionType &valueFunction,
    policy::FinitePolicyValueFunctionMixin<typename T::ValueFunctionType> &policy,
    policy::FinitePolicyValueFunctionMixin<typename T::ValueFunctionType> &target_policy,
    typename T::EnvironmentType &environment,
    const typename T::ActionSpace &action,
    const typename T::PrecisionType &discountRate) {

  typename T::ValueFunctionType;
  typename T::EnvironmentType;
  typename T::KeyType;
  typename T::KeyMaker;
  typename T::ValueType;
  typename T::PrecisionType;
  typename T::StateType;
  typename T::ActionSpace;
  typename T::StatefulUpdateResult;

  // Initialise both the environment and the value function. It may be uneccessary to initialise the value
  // function because it will be automatically initialised on the first update.
  t.initialize(environment, valueFunction);
  t.update(valueFunction, policy, target_policy, environment, action, discountRate);
};

/// @brief Value update interface. Value updates are responsible for modifying the value function during
/// episodic monte carlo value estimation. They may have internal state - such is a list of returns or
/// importance sampling rations.
template <typename T>
concept isTDValueUpdater = isTDUpdaterStep<T> && isTDUpdaterValueUpdate<T> && isTDUpdaterUpdate<T>;

template <typename CRTP>
struct DefaultValueUpdater {

  SETUP_TYPES_W_VALUE_FUNCTION(CRTP::ValueFunctionType);

  void updateValue(
      typename CRTP::ValueFunctionType &valueFunction,
      policy::isFinitePolicyValueFunctionMixin auto &policy,
      policy::isFinitePolicyValueFunctionMixin auto &target_policy,
      typename CRTP::EnvironmentType &environment,
      const typename CRTP::KeyType &keyCurrent,
      const typename CRTP::KeyType &keyNext,
      const typename CRTP::PrecisionType &reward,
      const typename CRTP::PrecisionType &discountRate) {

    valueFunction[keyCurrent].value =
        valueFunction.valueAt(keyCurrent) +
        temporal_differenc_error(
            valueFunction.valueAt(keyCurrent), valueFunction.valueAt(keyNext), reward, discountRate);
    valueFunction[keyCurrent].step++;
  };
};

template <policy::objectives::isFiniteStateValueFunction VALUE_FUNCTION_T>
struct TemporalDifferenceValueUpdaterBase {

  SETUP_TYPES_W_VALUE_FUNCTION(VALUE_FUNCTION_T);

  struct StatefulUpdateResult {
    bool isDone;
    ActionSpace action;
    TransitionType transition;
    PrecisionType reward;
  };
};

template <
    policy::objectives::isFiniteStateValueFunction VALUE_FUNCTION_T,
    template <typename CRTP>
    class StepInterface,
    template <typename CRTP> class ValueUpdaterInterface = DefaultValueUpdater>
struct TemporalDifferenceValueUpdater : StepInterface<TemporalDifferenceValueUpdaterBase<VALUE_FUNCTION_T>>,
                                        ValueUpdaterInterface<TemporalDifferenceValueUpdaterBase<VALUE_FUNCTION_T>> {

  static_assert(isTDUpdaterStep<StepInterface<TemporalDifferenceValueUpdaterBase<VALUE_FUNCTION_T>>>);
  static_assert(isTDUpdaterValueUpdate<ValueUpdaterInterface<TemporalDifferenceValueUpdaterBase<VALUE_FUNCTION_T>>>);

  SETUP_TYPES_W_VALUE_FUNCTION(VALUE_FUNCTION_T);

  using StepI = StepInterface<TemporalDifferenceValueUpdaterBase<VALUE_FUNCTION_T>>;
  using ValueI = ValueUpdaterInterface<TemporalDifferenceValueUpdaterBase<VALUE_FUNCTION_T>>;

  void initialize(typename VALUE_FUNCTION_T::EnvironmentType &environment, VALUE_FUNCTION_T &valueFunction) {
    valueFunction.initialize(environment);
  }

  std::pair<bool, ActionSpace> update(
      VALUE_FUNCTION_T &valueFunction,
      policy::isFinitePolicyValueFunctionMixin auto &policy,
      policy::isFinitePolicyValueFunctionMixin auto &target_policy,
      EnvironmentType &environment,
      const ActionSpace &action,
      const PrecisionType &discountRate);
};

template <
    policy::objectives::isFiniteStateValueFunction V,
    template <typename CRTP>
    class StepInterface,
    template <typename CRTP>
    class ValueUpdaterInterface>
auto TemporalDifferenceValueUpdater<V, StepInterface, ValueUpdaterInterface>::update(
    V &valueFunction,
    policy::isFinitePolicyValueFunctionMixin auto &policy,
    policy::isFinitePolicyValueFunctionMixin auto &target_policy,
    EnvironmentType &environment,
    const ActionSpace &action,
    const PrecisionType &discountRate) -> std::pair<bool, ActionSpace> {

  const auto [isDone, nextAction, transition, reward] =
      this->step(valueFunction, policy, target_policy, environment, action, discountRate);

  // Update the value function.
  this->updateValue(
      valueFunction,
      policy,
      target_policy,
      environment,
      KeyMaker::make(environment, transition.state, transition.action),
      KeyMaker::make(environment, transition.nextState, nextAction),
      reward,
      discountRate);

  return {transition.isDone(), nextAction};
}

} // namespace temporal_difference