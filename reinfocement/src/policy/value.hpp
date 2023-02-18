#pragma once

#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "environment.hpp"
#include "policy/objectives/value.hpp"
#include "policy/objectives/value_function.hpp"
#include "policy/objectives/value_function_keymaker.hpp"

#define PVT PolicyValueFunctionMixin<KEYMAPPER_T, VALUE_T, INITIAL_VALUE, DISCOUNT_RATE>

namespace policy {

template <objectives::isValueFunctionKeymaker KEYMAPPER_T,
          objectives::isValue VALUE_T,
          auto INITIAL_VALUE = 0.0F,
          auto DISCOUNT_RATE = 0.0F>
requires std::is_same_v<typename KEYMAPPER_T::EnvironmentType, typename VALUE_T::EnvironmentType>
struct PolicyValueFunctionMixin
    : virtual objectives::ValueFunction<KEYMAPPER_T, VALUE_T, INITIAL_VALUE, DISCOUNT_RATE> {

  using BaseType = objectives::ValueFunction<KEYMAPPER_T, VALUE_T, INITIAL_VALUE, DISCOUNT_RATE>;
  using ValueFunctionType = objectives::ValueFunction<KEYMAPPER_T, VALUE_T, INITIAL_VALUE, DISCOUNT_RATE>;
  using KeyMaker = KEYMAPPER_T;
  using ValueType = VALUE_T;
  SETUP_TYPES_FROM_NESTED_ENVIRON(SINGLE_ARG(BaseType::EnvironmentType));

  virtual PrecisionType getValue(const EnvironmentType &e, const StateType &s, const ActionSpace &a) const;
  // virtual PrecisionType getValue(const EnvironmentType &e, const StateType &s) const = 0;
  // virtual ActionSpace getArgmaxAction(const EnvironmentType &e, const StateType &s) const = 0;
};

template <objectives::isValueFunctionKeymaker KEYMAPPER_T,
          objectives::isValue VALUE_T,
          auto INITIAL_VALUE,
          auto DISCOUNT_RATE>
typename PVT::PrecisionType PVT::getValue(const EnvironmentType &e, const StateType &s, const ActionSpace &a) const {
  return this->valueAt(this->makeKey(e, s, a));
}

template <typename T>
concept isPolicyValueFunctionMixin = std::is_base_of_v<
    PolicyValueFunctionMixin<typename T::KeyMaker, typename T::ValueType, T::initial_value, T::discount_rate>,
    T>;

} // namespace policy

#undef PVT
