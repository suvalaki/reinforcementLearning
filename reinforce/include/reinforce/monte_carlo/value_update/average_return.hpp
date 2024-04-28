#pragma once

#include <unordered_map>
#include <vector>

#include "reinforce/monte_carlo/episode.hpp"
#include "reinforce/monte_carlo/value_update/value_update.hpp"
#include "reinforce/policy/distribution_policy.hpp"
#include "reinforce/policy/finite/value_policy.hpp"
#include "reinforce/policy/objectives/finite_value_function.hpp"
#include "reinforce/policy/value.hpp"

namespace monte_carlo {

template <policy::objectives::isFiniteStateValueFunction VALUE_FUNCTION_T>
struct NiaveAverageReturnsUpdate : ValueUpdaterBase<NiaveAverageReturnsUpdate<VALUE_FUNCTION_T>, VALUE_FUNCTION_T> {

  SETUP_TYPES_W_VALUE_FUNCTION(VALUE_FUNCTION_T);
  using ReturnsContainer = std::vector<typename VALUE_FUNCTION_T::PrecisionType>;
  using ReturnsMap = std::
      unordered_map<typename VALUE_FUNCTION_T::KeyType, ReturnsContainer, typename VALUE_FUNCTION_T::KeyMaker::Hash>;

  ReturnsMap returns;

  void updateReturns(
      VALUE_FUNCTION_T &valueFunction,
      policy::isFinitePolicyValueFunctionMixin auto &policy,
      policy::isFinitePolicyValueFunctionMixin auto &target_policy,
      typename VALUE_FUNCTION_T::EnvironmentType &environment,
      const StateType &state,
      const ActionSpace &action,
      const typename VALUE_FUNCTION_T::PrecisionType &discountedReturn) {
    const auto key = KeyMaker::make(environment, state, action);
    returns[key].emplace_back(discountedReturn);
  }

  PrecisionType getAverageReturn(const KeyType &key) {
    auto &ret = returns[key];
    PrecisionType sum = 0;
    for (const auto &r : ret) {
      sum += r;
    }
    return sum / ret.size();
  }

  void updateValue(
      VALUE_FUNCTION_T &valueFunction,
      policy::isFinitePolicyValueFunctionMixin auto &policy,
      policy::isFinitePolicyValueFunctionMixin auto &target_policy,
      typename VALUE_FUNCTION_T::EnvironmentType &environment,
      const StateType &state,
      const ActionSpace &action) {
    const auto key = KeyMaker::make(environment, state, action);
    valueFunction[key].value = getAverageReturn(key);
    valueFunction[key].step = returns[key].size();
  }
};

template <policy::objectives::isFiniteStateValueFunction VALUE_FUNCTION_T>
struct NiaveAverageReturnsIncrementalUpdate
    : ValueUpdaterBase<NiaveAverageReturnsIncrementalUpdate<VALUE_FUNCTION_T>, VALUE_FUNCTION_T> {

  SETUP_TYPES_W_VALUE_FUNCTION(VALUE_FUNCTION_T);

  struct ReturnsContainer {
    typename VALUE_FUNCTION_T::PrecisionType averageReturn = 0;
    size_t n = 0;
  };

  using ReturnsMap = std::
      unordered_map<typename VALUE_FUNCTION_T::KeyType, ReturnsContainer, typename VALUE_FUNCTION_T::KeyMaker::Hash>;

  ReturnsMap returns;

  void updateReturns(
      VALUE_FUNCTION_T &valueFunction,
      policy::isFinitePolicyValueFunctionMixin auto &policy,
      policy::isFinitePolicyValueFunctionMixin auto &target_policy,
      typename VALUE_FUNCTION_T::EnvironmentType &environment,
      const StateType &state,
      const ActionSpace &action,
      const typename VALUE_FUNCTION_T::PrecisionType &discountedReturns) {
    const auto key = KeyMaker::make(environment, state, action);
    returns[key].averageReturn =
        (returns[key].averageReturn * returns[key].n + discountedReturns) / (returns[key].n + 1);
    returns[key].n++;
  }

  void updateValue(
      VALUE_FUNCTION_T &valueFunction,
      policy::isFinitePolicyValueFunctionMixin auto &policy,
      policy::isFinitePolicyValueFunctionMixin auto &target_policy,
      typename VALUE_FUNCTION_T::EnvironmentType &environment,
      const StateType &state,
      const ActionSpace &action) {
    const auto key = KeyMaker::make(environment, state, action);
    valueFunction[key].value = returns[key].averageReturn;
    valueFunction[key].step++;
  }
};

} // namespace monte_carlo
