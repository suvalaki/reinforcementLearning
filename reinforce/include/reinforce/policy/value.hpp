#pragma once

#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "reinforce/environment.hpp"
#include "reinforce/policy/objectives/finite_value_function_combination.hpp"
#include "reinforce/policy/objectives/value.hpp"
#include "reinforce/policy/objectives/value_function.hpp"
#include "reinforce/policy/objectives/value_function_keymaker.hpp"

#define PVT PolicyValueFunctionMixin<VALUE_FUNCTION_T>

namespace policy {

template <objectives::isValueFunction VALUE_FUNCTION_T>
struct PolicyValueFunctionMixin : VALUE_FUNCTION_T {

  using BaseType = VALUE_FUNCTION_T;
  SETUP_TYPES_FROM_NESTED_ENVIRON(SINGLE_ARG(BaseType::EnvironmentType));
  using ValueFunctionType = VALUE_FUNCTION_T;
  using KeyMaker = typename VALUE_FUNCTION_T::KeyMaker;
  using ValueType = typename VALUE_FUNCTION_T::ValueType;

  PolicyValueFunctionMixin(auto &&...args) : VALUE_FUNCTION_T(args...) {}
  PolicyValueFunctionMixin(const PolicyValueFunctionMixin &p) : VALUE_FUNCTION_T(p) {}

  virtual PrecisionType getValue(const EnvironmentType &e, const StateType &s, const ActionSpace &a);
};

template <objectives::isValueFunction VALUE_FUNCTION_T>
typename PVT::PrecisionType PVT::getValue(const EnvironmentType &e, const StateType &s, const ActionSpace &a) {
  return this->valueAt(this->makeKey(e, s, a));
}

template <typename T>
concept isPolicyValueFunctionMixin = std::is_base_of_v<PolicyValueFunctionMixin<typename T::BaseType>, T>;

} // namespace policy

#undef PVT
