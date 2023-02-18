#pragma once

#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "environment.hpp"
#include "policy/finite/policy.hpp"
#include "policy/objectives/finite_value.hpp"
#include "policy/objectives/finite_value_function.hpp"
#include "policy/objectives/value_function_keymaker.hpp"
#include "policy/value.hpp"

#define FVT FinitePolicyValueFunctionMixin<BASE_VALUEFUNCTION_POLICY_T, INCREMENTAL_STEPSIZE_T>

namespace policy {

// The distinguishing feature of a finite value policy is the ability to hold mappins from
// all of the keys in the environment to values.

template <isPolicyValueFunctionMixin BASE_VALUEFUNCTION_POLICY_T,
          objectives::isStepSizeTaker INCREMENTAL_STEPSIZE_T =
              objectives::weighted_average_step_size_taker<typename BASE_VALUEFUNCTION_POLICY_T::ValueType>>
// requires std::is_same_v<typename KEYMAPPER_T::EnvironmentType, typename VALUE_T::EnvironmentType>
struct FinitePolicyValueFunctionMixin
    : virtual PolicyDistributionMixin<typename BASE_VALUEFUNCTION_POLICY_T::EnvironmentType>,
      virtual objectives::FiniteValueFunction<BASE_VALUEFUNCTION_POLICY_T, INCREMENTAL_STEPSIZE_T> {

  using BaseType = BASE_VALUEFUNCTION_POLICY_T;
  SETUP_TYPES_FROM_NESTED_ENVIRON(SINGLE_ARG(BASE_VALUEFUNCTION_POLICY_T::EnvironmentType));
  using ValueFunctionType = objectives::FiniteValueFunction<BASE_VALUEFUNCTION_POLICY_T, INCREMENTAL_STEPSIZE_T>;
  using KeyMaker = typename ValueFunctionType::KeyMaker;
  using ValueType = typename ValueFunctionType::ValueType;
  using StepSizeTaker = typename ValueFunctionType::StepSizeTaker;

  ActionSpace getArgmaxAction(const EnvironmentType &e, const StateType &s) const override;
};

template <isPolicyValueFunctionMixin BASE_VALUEFUNCTION_POLICY_T, objectives::isStepSizeTaker INCREMENTAL_STEPSIZE_T>
typename FVT::ActionSpace FVT::getArgmaxAction(const EnvironmentType &e, const StateType &s) const {

  auto action = ActionSpace{}; // start with a random action so we at least have one that is permissible
  auto maxIdx = std::max_element(
      this->begin(), this->end(), [](const auto &p1, const auto &p2) { return p1.second < p2.second; });
  if (maxIdx != this->end()) {
    action = KeyMaker::get_action_from_key(e, maxIdx->first);
  }

  return action;
}

template <typename T>
concept isFinitePolicyValueFunctionMixin =
    std::is_base_of_v<FinitePolicyValueFunctionMixin<typename T::ValueFunctionBaseType, typename T::StepSizeTaker>, T>;

template <typename T>
concept implementsFiniteValuePolicy = isPolicyValueFunctionMixin<T> && isFinitePolicy<T>;

} // namespace policy

#undef FVT