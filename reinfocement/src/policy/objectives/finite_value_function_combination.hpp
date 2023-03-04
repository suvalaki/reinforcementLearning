#pragma once
#include <type_traits>

#include "policy/objectives/finite_value_function.hpp"
#include "policy/objectives/value_function_combination.hpp"

namespace policy::objectives {

template <isFiniteStateValueFunction... VALUE_FUNCTION_T> struct is_finite_value_function_check;
template <isFiniteStateValueFunction VALUE_FUNCTION_T>
struct is_finite_value_function_check<VALUE_FUNCTION_T> : std::true_type {};
template <isFiniteStateValueFunction VALUE_FUNCTION_T,
          isFiniteStateValueFunction VALUE_FUNCTION_T2,
          isFiniteStateValueFunction... VALUE_FUNCTION_Ts>
struct is_finite_value_function_check<VALUE_FUNCTION_T, VALUE_FUNCTION_T2, VALUE_FUNCTION_Ts...>
    : std::bool_constant<std::is_same_v<typename VALUE_FUNCTION_T::ValueType, typename VALUE_FUNCTION_T2::ValueType> &&
                         is_finite_value_function_check<VALUE_FUNCTION_T2, VALUE_FUNCTION_Ts...>::value> {};

template <typename... VALUE_FUNCTION_T>
concept isAdmissibleFiniteStateValueFunctionCombination = is_finite_value_function_check<VALUE_FUNCTION_T...>::value;

/// @brief Get a generic Value Function from the first element of the parameter pack
template <isValueFunction... VALUE_FUNCTION_T>
requires isAdmissibleFiniteStateValueFunctionCombination<VALUE_FUNCTION_T...>
struct get_first_finite_value_function_type_generic {
  using vFunctType = typename get_first_value_function_type<VALUE_FUNCTION_T...>::type;
  using type = FiniteValueFunction<vFunctType, typename vFunctType::StepSizeTaker>;
};

template <isFiniteStateValueFunction... VALUE_FUNCTION_T>
requires is_finite_value_function_check<VALUE_FUNCTION_T...>::value struct AdditiveFiniteValueFunctionCombination
    : virtual AdditiveValueFunctionCombination<VALUE_FUNCTION_T...>,
      virtual get_first_finite_value_function_type_generic<VALUE_FUNCTION_T...>::type {

  using BaseType = AdditiveValueFunctionCombination<VALUE_FUNCTION_T...>;
  using AdditiveValueFunctionCombination<VALUE_FUNCTION_T...>::AdditiveValueFunctionCombination;
  using fValueFunctionType = get_first_finite_value_function_type_generic<VALUE_FUNCTION_T...>::type;
  using ValueType = typename fValueFunctionType::ValueType;
  using KeyType = typename fValueFunctionType::KeyType;

  SETUP_TYPES_FROM_NESTED_ENVIRON(SINGLE_ARG(BaseType::EnvironmentType));

  ValueType operator()(const KeyType &k) const override { return BaseType::operator()(k); }
  using BaseType::initialize;

  // We intercept the call to the value function and return the sum of the values returned by each value function.
};

} // namespace policy::objectives
