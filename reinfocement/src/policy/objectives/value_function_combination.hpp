#pragma once
#include <tuple>
#include <type_traits>

#include "policy/objectives/value_function.hpp"

namespace policy::objectives {

/// @brief Validate that the Value type being returned by every value function is the same
template <isValueFunction... VALUE_FUNCTION_T> struct value_function_admissible_impl;
template <isValueFunction VALUE_FUNCTION_T> struct value_function_admissible_impl<VALUE_FUNCTION_T> : std::true_type {};
template <isValueFunction VALUE_FUNCTION_T, isValueFunction VALUE_FUNCTION_T2, isValueFunction... VALUE_FUNCTION_Ts>
struct value_function_admissible_impl<VALUE_FUNCTION_T, VALUE_FUNCTION_T2, VALUE_FUNCTION_Ts...>
    : std::bool_constant<std::is_same_v<typename VALUE_FUNCTION_T::ValueType, typename VALUE_FUNCTION_T2::ValueType> &&
                         value_function_admissible_impl<VALUE_FUNCTION_T2, VALUE_FUNCTION_Ts...>::value> {};

template <typename... VALUE_FUNCTION_T>
concept isAdmissibleValueFunctionCombination = value_function_admissible_impl<VALUE_FUNCTION_T...>::value;

template <isValueFunction... VALUE_FUNCTION_T>
requires isAdmissibleValueFunctionCombination<VALUE_FUNCTION_T...>
struct get_first_value_function_type {
  using type = std::tuple_element_t<0, std::tuple<VALUE_FUNCTION_T...>>;
};
/// @brief Get a generic Value Function from the first element of the parameter pack
template <isValueFunction... VALUE_FUNCTION_T>
requires isAdmissibleValueFunctionCombination<VALUE_FUNCTION_T...>
struct get_first_value_function_type_generic {
  using vFunctType = typename get_first_value_function_type<VALUE_FUNCTION_T...>::type;
  using type = ValueFunction<typename vFunctType::KeyMaker,
                             typename vFunctType::ValueType,
                             vFunctType::initial_value,
                             vFunctType::discount_rate>;
};

/// @brief A base template interface for use in defining different types of combinations of policies.
template <isValueFunction... VALUE_FUNCTION_T>
requires isAdmissibleValueFunctionCombination<VALUE_FUNCTION_T...>
struct ValueFunctionCombination : get_first_value_function_type_generic<VALUE_FUNCTION_T...>::type {

  using ValueFunctionTypes = std::tuple<VALUE_FUNCTION_T...>;
  SETUP_TYPES_FROM_NESTED_ENVIRON(SINGLE_ARG(std::tuple_element_t<0, ValueFunctionTypes>::EnvironmentType));
  using ValueType = std::tuple_element_t<0, ValueFunctionTypes>::ValueType;
  using KeyType = std::tuple_element_t<0, ValueFunctionTypes>::KeyType;

  using ValueFunctionRefs = std::tuple<VALUE_FUNCTION_T &...>;
  ValueFunctionRefs valueFunctions;

  ValueFunctionCombination(VALUE_FUNCTION_T &...valueFunctions) : valueFunctions(valueFunctions...) {}
  ValueFunctionCombination(const ValueFunctionCombination &v) : valueFunctions(v.valueFunctions) {}

  void initialize(EnvironmentType &environment) override {
    std::apply([&](auto &...valueFunctions) { (valueFunctions.initialize(environment), ...); }, this->valueFunctions);
  }

  // How the combination is applied is an implementation detail for the specific combination.
};

/// @brief A combination of value functions that returns the sum of the values returned by each value function.
/// @tparam ...VALUE_FUNCTION_T
template <isValueFunction... VALUE_FUNCTION_T>
requires isAdmissibleValueFunctionCombination<VALUE_FUNCTION_T...>
struct AdditiveValueFunctionCombination : ValueFunctionCombination<VALUE_FUNCTION_T...> {

  using BaseType = ValueFunctionCombination<VALUE_FUNCTION_T...>;
  AdditiveValueFunctionCombination(auto &&...args) : ValueFunctionCombination<VALUE_FUNCTION_T...>(args...) {}

  SETUP_TYPES_FROM_NESTED_ENVIRON(SINGLE_ARG(BaseType::EnvironmentType));
  using ValueType = typename BaseType::ValueType;
  using KeyType = typename BaseType::KeyType;

  ValueType operator()(const KeyType &k) const override;
  // PrecisionType operator()(const KeyType &k, const ValueType &v) const override;
};

template <isValueFunction... VALUE_FUNCTION_T>
auto AdditiveValueFunctionCombination<VALUE_FUNCTION_T...>::operator()(const KeyType &k) const ->
    typename AdditiveValueFunctionCombination<VALUE_FUNCTION_T...>::ValueType {
  ValueType result = {};
  std::apply([&](auto &...valueFunctions) { ((result += valueFunctions[k]), ...); }, this->valueFunctions);
  return result;
}

// template <isValueFunction... VALUE_FUNCTION_T>
// auto AdditiveValueFunctionCombination<VALUE_FUNCTION_T...>::operator()(const KeyType &k, const ValueType &v) const ->
//     typename AdditiveValueFunctionCombination<VALUE_FUNCTION_T...>::PrecisionType {
//   PrecisionType result = 0;
//   std::apply([&](auto &...valueFunctions) { ((result += valueFunctions(k, v)), ...); }, this->valueFunctions);
//   return result;
// }

} // namespace policy::objectives
