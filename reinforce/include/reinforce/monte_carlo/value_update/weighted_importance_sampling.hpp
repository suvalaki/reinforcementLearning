#pragma once

#include <unordered_map>
#include <vector>

#include "reinforce/monte_carlo/episode.hpp"
#include "reinforce/monte_carlo/value_update/ordinary_importance_sampling.hpp"
#include "reinforce/monte_carlo/value_update/value_update.hpp"
#include "reinforce/policy/distribution_policy.hpp"
#include "reinforce/policy/finite/value_policy.hpp"
#include "reinforce/policy/objectives/finite_value_function.hpp"
#include "reinforce/policy/value.hpp"

namespace monte_carlo {

template <policy::objectives::isFiniteStateValueFunction VALUE_FUNCTION_T>
struct WeightedImportanceSamplingUpdate
    : OrdinaryImportanceSamplingUpdate<VALUE_FUNCTION_T>,
      ValueUpdaterBase<WeightedImportanceSamplingUpdate<VALUE_FUNCTION_T>, VALUE_FUNCTION_T> {

  SETUP_TYPES_W_VALUE_FUNCTION(VALUE_FUNCTION_T);

  virtual PrecisionType getWeightedReturn(const KeyType &key) {
    auto &ret = this->returns[key];
    PrecisionType sum = 0;
    PrecisionType sum_importance_sampling_ratio = 0;
    for (const auto &r : ret) {
      sum += r.ret * r.importance_sampling_ratio;
      sum_importance_sampling_ratio += r.importance_sampling_ratio;
    }
    return sum / sum_importance_sampling_ratio;
  }

  template <policy::isFinitePolicyValueFunctionMixin POLICY_T0, policy::isFinitePolicyValueFunctionMixin POLICY_T1>
  void updateValue(
      VALUE_FUNCTION_T &valueFunction,
      POLICY_T0 &policy,
      POLICY_T1 &target_policy,
      typename VALUE_FUNCTION_T::EnvironmentType &environment,
      const StateType &state,
      const ActionSpace &action) {
    const auto key = KeyMaker::make(environment, state, action);
    valueFunction[key].value = getWeightedReturn(key);
    valueFunction[key].step = this->returns[key].size();
  }
};

template <policy::objectives::isFiniteStateValueFunction VALUE_FUNCTION_T>
struct WeightedImportanceSamplingIncrementalUpdate
    : ValueUpdaterBase<NiaveAverageReturnsIncrementalUpdate<VALUE_FUNCTION_T>, VALUE_FUNCTION_T> {

  SETUP_TYPES_W_VALUE_FUNCTION(VALUE_FUNCTION_T);

  struct ReturnsContainer {
    typename VALUE_FUNCTION_T::PrecisionType averageWeightedReturn = 0;
    typename VALUE_FUNCTION_T::PrecisionType importanceWeight = 1.0F;
    typename VALUE_FUNCTION_T::PrecisionType cumulativeImportanceWeight = 0;
    size_t n = 0;
  };

  using ReturnsMap = std::
      unordered_map<typename VALUE_FUNCTION_T::KeyType, ReturnsContainer, typename VALUE_FUNCTION_T::KeyMaker::Hash>;

  ReturnsMap returns;

  template <policy::isFinitePolicyValueFunctionMixin POLICY_T0, policy::isFinitePolicyValueFunctionMixin POLICY_T1>
  void updateReturns(
      VALUE_FUNCTION_T &valueFunction,
      POLICY_T0 &policy,
      POLICY_T1 &target_policy,
      typename VALUE_FUNCTION_T::EnvironmentType &environment,
      const StateType &state,
      const ActionSpace &action,
      const typename VALUE_FUNCTION_T::PrecisionType &discountedReturns) {
    const auto key = KeyMaker::make(environment, state, action);

    const auto importanceRatio = target_policy.importanceSamplingRatio(environment, state, action, policy);
    if (returns.find(key) == returns.end()) {
      returns[key] = ReturnsContainer(discountedReturns, importanceRatio, importanceRatio, 2);
    } else {
      const auto previousImportanceWeight = returns[key].importanceWeight;
      const auto newImportanceWeight = previousImportanceWeight * importanceRatio;
      returns[key].averageWeightedReturn =
          (returns[key].averageWeightedReturn * returns[key].cumulativeImportanceWeight +
           newImportanceWeight * discountedReturns) /
          (returns[key].cumulativeImportanceWeight + newImportanceWeight);
      returns[key].importanceWeight *= importanceRatio;
      returns[key].cumulativeImportanceWeight += returns[key].importanceWeight;
      returns[key].n++;
    }
  }

  template <policy::isFinitePolicyValueFunctionMixin POLICY_T0, policy::isFinitePolicyValueFunctionMixin POLICY_T1>
  void updateValue(
      VALUE_FUNCTION_T &valueFunction,
      POLICY_T0 &policy,
      POLICY_T1 &target_policy,
      typename VALUE_FUNCTION_T::EnvironmentType &environment,
      const StateType &state,
      const ActionSpace &action) {
    const auto key = KeyMaker::make(environment, state, action);
    valueFunction[key].value = returns[key].averageWeightedReturn;
    valueFunction[key].step++;
  }
};

} // namespace monte_carlo
