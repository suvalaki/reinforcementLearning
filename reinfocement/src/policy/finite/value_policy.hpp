#pragma once

#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "environment.hpp"
#include "policy/combination_policy.hpp"
#include "policy/finite/policy.hpp"
#include "policy/objectives/finite_value.hpp"
#include "policy/objectives/finite_value_function.hpp"
#include "policy/objectives/value_function_keymaker.hpp"
#include "policy/value.hpp"

#define FVT FinitePolicyValueFunctionMixin<VALUE_FUNCTION_T>

namespace policy {

// The distinguishing feature of a finite value policy is the ability to hold mappins from
// all of the keys in the environment to values.

template <objectives::isFiniteValueFunction VALUE_FUNCTION_T>
// requires std::is_same_v<typename KEYMAPPER_T::EnvironmentType, typename VALUE_T::EnvironmentType>
struct FinitePolicyValueFunctionMixin : virtual PolicyDistributionMixin<typename VALUE_FUNCTION_T::EnvironmentType>,
                                        virtual PolicyValueFunctionMixin<VALUE_FUNCTION_T>,
                                        virtual VALUE_FUNCTION_T {

  using BaseType = VALUE_FUNCTION_T;
  SETUP_TYPES_FROM_NESTED_ENVIRON(SINGLE_ARG(VALUE_FUNCTION_T::EnvironmentType));
  using ValueFunctionType = VALUE_FUNCTION_T;
  using ValueFunctionBaseType = typename ValueFunctionType::ValueFunctionBaseType;
  using KeyMaker = typename ValueFunctionType::KeyMaker;
  using ValueType = typename ValueFunctionType::ValueType;
  using StepSizeTaker = typename ValueFunctionType::StepSizeTaker;

  ActionSpace getArgmaxAction(const EnvironmentType &e, const StateType &s) const override;
  using ValueFunctionType::initialize;
};

template <objectives::isFiniteValueFunction VALUE_FUNCTION_T>
typename FVT::ActionSpace FVT::getArgmaxAction(const EnvironmentType &e, const StateType &s) const {
  return KeyMaker::get_action_from_key(e, this->getArgmaxKey(e, s));
}

template <typename T>
concept isFinitePolicyValueFunctionMixin =
    std::is_base_of_v<FinitePolicyValueFunctionMixin<typename T::ValueFunctionType>, T>;

template <typename T>
concept implementsFiniteValuePolicy = isFinitePolicyValueFunctionMixin<T> && isFinitePolicy<T>;

} // namespace policy

#undef FVT