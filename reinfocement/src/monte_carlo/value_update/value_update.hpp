#pragma once

#include <unordered_map>
#include <vector>

#include "monte_carlo/episode.hpp"
#include "policy/distribution_policy.hpp"
#include "policy/finite/greedy_policy.hpp"
#include "policy/finite/value_policy.hpp"
#include "policy/objectives/finite_value_function.hpp"
#include "policy/value.hpp"

namespace monte_carlo {

/// @brief Value update interface. Value updates are responsible for modifying the value function during
/// episodic monte carlo value estimation. They may have internal state - such is a list of returns or
/// importance sampling rations.
template <typename T>
concept isValueUpdater = requires(T t) {

  typename T::ValueFunctionType;
  typename T::EnvironmentType;
  typename T::KeyType;
  typename T::KeyMaker;
  typename T::ValueType;
  typename T::PrecisionType;
  typename T::StateType;
  typename T::ActionSpace;
  // typename T::ReturnsContainer;

  // Initialise both the environment and the value function. It may be uneccessary to initialise the value
  // function because it will be automatically initialised on the first update.
  t.initialize(std::declval<typename T::EnvironmentType &>(), std::declval<typename T::ValueFunctionType &>());

  // Update the internal state of the Updater itself. When the Updater sees a new return it can perform
  // calculations like weighted averaging, incrementing support or importance sampling estimation.
  t.updateReturns(std::declval<typename T::ValueFunctionType &>(),
                  std::declval<policy::FiniteGreedyPolicy<typename T::KeyMaker, typename T::ValueType> &>(),
                  std::declval<policy::FiniteGreedyPolicy<typename T::KeyMaker, typename T::ValueType> &>(),
                  std::declval<typename T::EnvironmentType &>(),
                  std::declval<typename T::StateType &>(),
                  std::declval<typename T::ActionSpace &>(),
                  std::declval<typename T::PrecisionType &>());

  // Using the newly updated state of the Updater, update the value function.
  t.updateValue(std::declval<typename T::ValueFunctionType &>(),
                std::declval<policy::FiniteGreedyPolicy<typename T::KeyMaker, typename T::ValueType> &>(),
                std::declval<policy::FiniteGreedyPolicy<typename T::KeyMaker, typename T::ValueType> &>(),
                std::declval<typename T::EnvironmentType &>(),
                std::declval<typename T::StateType &>(),
                std::declval<typename T::ActionSpace &>());

  // Update the internal state of the Updater and then the value function.
  t.update(std::declval<typename T::ValueFunctionType &>(),
           std::declval<policy::FiniteGreedyPolicy<typename T::KeyMaker, typename T::ValueType> &>(),
           std::declval<policy::FiniteGreedyPolicy<typename T::KeyMaker, typename T::ValueType> &>(),
           std::declval<typename T::EnvironmentType &>(),
           std::declval<typename T::StateType &>(),
           std::declval<typename T::ActionSpace &>(),
           std::declval<typename T::PrecisionType &>());
};

template <typename CRTP, policy::objectives::isFiniteStateValueFunction VALUE_FUNCTION_T> struct ValueUpdaterBase {

  void initialize(typename VALUE_FUNCTION_T::EnvironmentType &environment, VALUE_FUNCTION_T &valueFunction) {
    valueFunction.initialize(environment);
    // for (const auto &s : environment.getAllPossibleStates()) {
    //   for (const auto &a : environment.getReachableActions(s)) {
    //     static_cast<CRTP &>(*this).returns[typename VALUE_FUNCTION_T::KeyMaker::make(environment, s, a)] =
    //         ReturnsContainer();
    //   }
    // }
  }

  template <policy::isFinitePolicyValueFunctionMixin POLICY_T0, policy::isFinitePolicyValueFunctionMixin POLICY_T1>
  void update(VALUE_FUNCTION_T &valueFunction,
              POLICY_T0 &policy,
              POLICY_T1 &target_policy,
              typename VALUE_FUNCTION_T::EnvironmentType &environment,
              const typename VALUE_FUNCTION_T::StateType &state,
              const typename VALUE_FUNCTION_T::ActionSpace &action,
              const typename VALUE_FUNCTION_T::PrecisionType &discountedReturn) {
    static_cast<CRTP &>(*this).updateReturns(
        valueFunction, policy, target_policy, environment, state, action, discountedReturn);
    static_cast<CRTP &>(*this).updateValue(valueFunction, policy, target_policy, environment, state, action);
  }
};

} // namespace monte_carlo