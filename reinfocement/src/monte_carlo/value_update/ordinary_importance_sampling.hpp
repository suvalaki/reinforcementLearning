#pragma once

#include <unordered_map>
#include <vector>

#include "monte_carlo/episode.hpp"
#include "monte_carlo/value_update/value_update.hpp"
#include "policy/distribution_policy.hpp"
#include "policy/finite/value_policy.hpp"
#include "policy/objectives/finite_value_function.hpp"
#include "policy/value.hpp"

namespace monte_carlo {

template <policy::objectives::isFiniteStateValueFunction VALUE_FUNCTION_T>
struct OrdinaryImportanceSamplingUpdate
    : ValueUpdaterBase<NiaveAverageReturnsUpdate<VALUE_FUNCTION_T>, VALUE_FUNCTION_T> {

  SETUP_TYPES_FROM_NESTED_ENVIRON(SINGLE_ARG(VALUE_FUNCTION_T::EnvironmentType));

  using ValueFunctionType = VALUE_FUNCTION_T;
  using KeyMaker = typename VALUE_FUNCTION_T::KeyMaker;
  using KeyType = typename VALUE_FUNCTION_T::KeyType;
  using ValueType = typename VALUE_FUNCTION_T::ValueType;

  struct ImportanceWeightedReturn {
    PrecisionType ret = 0;
    PrecisionType importance_sampling_ratio = 1;
  };

  using ReturnsContainer = std::vector<ImportanceWeightedReturn>;
  using ReturnsMap = std::
      unordered_map<typename VALUE_FUNCTION_T::KeyType, ReturnsContainer, typename VALUE_FUNCTION_T::KeyMaker::Hash>;

  ReturnsMap returns;

  template <policy::isFinitePolicyValueFunctionMixin POLICY_T0, policy::isFinitePolicyValueFunctionMixin POLICY_T1>
  void updateReturns(VALUE_FUNCTION_T &valueFunction,
                     POLICY_T0 &policy,
                     POLICY_T1 &target_policy,
                     typename VALUE_FUNCTION_T::EnvironmentType &environment,
                     const StateType &state,
                     const ActionSpace &action,
                     const typename VALUE_FUNCTION_T::PrecisionType &discountedReturn) {
    const auto key = KeyMaker::make(environment, state, action);
    const auto thisImportanceRatio = target_policy.importanceSamplingRatio(environment, state, action, policy);
    if (returns.find(key) == returns.end()) {
      returns[key].emplace_back(ImportanceWeightedReturn{discountedReturn, thisImportanceRatio});
    } else {
      auto prevRatio = returns[key].back().importance_sampling_ratio;
      returns[key].emplace_back(ImportanceWeightedReturn{discountedReturn, prevRatio * thisImportanceRatio});
    }
  }

  virtual PrecisionType getWeightedReturn(const KeyType &key) {
    if (returns.find(key) == returns.end()) {
      return 0;
    }
    auto &ret = returns[key];
    PrecisionType sum = 0;
    for (const auto &r : ret) {
      sum += r.ret * r.importance_sampling_ratio;
    }
    return sum / ret.size();
  }

  template <policy::isFinitePolicyValueFunctionMixin POLICY_T0, policy::isFinitePolicyValueFunctionMixin POLICY_T1>
  void updateValue(VALUE_FUNCTION_T &valueFunction,
                   POLICY_T0 &policy,
                   POLICY_T1 &target_policy,
                   typename VALUE_FUNCTION_T::EnvironmentType &environment,
                   const StateType &state,
                   const ActionSpace &action) {
    const auto key = KeyMaker::make(environment, state, action);
    valueFunction[key].value = getWeightedReturn(key);
    valueFunction[key].step = returns[key].size();
  }
};

template <policy::objectives::isFiniteStateValueFunction VALUE_FUNCTION_T>
struct OrdinaryImportanceSamplingIncrementalUpdate
    : ValueUpdaterBase<NiaveAverageReturnsIncrementalUpdate<VALUE_FUNCTION_T>, VALUE_FUNCTION_T> {

  SETUP_TYPES_FROM_NESTED_ENVIRON(SINGLE_ARG(VALUE_FUNCTION_T::EnvironmentType));
  using KeyMaker = typename VALUE_FUNCTION_T::KeyMaker;
  using KeyType = typename VALUE_FUNCTION_T::KeyType;

  struct ReturnsContainer {
    typename VALUE_FUNCTION_T::PrecisionType averageWeightedReturn = 0;
    PrecisionType importance_sampling_ratio = 1;
    size_t n = 0;
  };

  using ReturnsMap = std::
      unordered_map<typename VALUE_FUNCTION_T::KeyType, ReturnsContainer, typename VALUE_FUNCTION_T::KeyMaker::Hash>;

  ReturnsMap returns;

  template <policy::isFinitePolicyValueFunctionMixin POLICY_T0, policy::isFinitePolicyValueFunctionMixin POLICY_T1>
  void updateReturns(VALUE_FUNCTION_T &valueFunction,
                     POLICY_T0 &policy,
                     POLICY_T1 &target_policy,
                     typename VALUE_FUNCTION_T::EnvironmentType &environment,
                     const StateType &state,
                     const ActionSpace &action,
                     const typename VALUE_FUNCTION_T::PrecisionType &discountedReturns) {
    const auto key = KeyMaker::make(environment, state, action);
    const auto importanceRatio = target_policy.importanceSamplingRatio(environment, state, action, policy);
    const auto prevRatio = returns[key].importance_sampling_ratio;
    returns[key].averageWeightedReturn =
        (returns[key].averageWeightedReturn * returns[key].n + prevRatio * importanceRatio * discountedReturns) /
        (returns[key].n + 1);
    returns[key].importance_sampling_ratio *= importanceRatio;
    returns[key].n++;
  }

  template <policy::isFinitePolicyValueFunctionMixin POLICY_T0, policy::isFinitePolicyValueFunctionMixin POLICY_T1>
  void updateValue(VALUE_FUNCTION_T &valueFunction,
                   POLICY_T0 &policy,
                   POLICY_T1 &target_policy,
                   typename VALUE_FUNCTION_T::EnvironmentType &environment,
                   const StateType &state,
                   const ActionSpace &action) {
    const auto key = KeyMaker::make(environment, state, action);
    if (returns.find(key) == returns.end()) {
      return;
    }
    valueFunction[key].value = returns[key].averageWeightedReturn;
    valueFunction[key].step++;
  }
};

} // namespace monte_carlo