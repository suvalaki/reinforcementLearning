#pragma once

#include <tuple>
#include <type_traits>

#include "reinforce/policy/objectives/value_function.hpp"

namespace policy::objectives {

/** @brief Template to perform a check for whether value functions all share the same ValueType */
template <isValueFunction... T>
struct is_same_value_function_check;

template <isValueFunction T>
struct is_same_value_function_check<T> : std::true_type {};

template <isValueFunction T0, isValueFunction T1, isValueFunction... Tn>
struct is_same_value_function_check<T0, T1, Tn...>
    : std::bool_constant<
          std::is_same_v<typename T0::ValueType, typename T1::ValueType> &&
          is_same_value_function_check<T1, Tn...>::value> {};

/** @brief Template alias to perform a check for whether value functions all share the same ValueType */
template <typename... T>
concept is_same_value_function_check_v = is_same_value_function_check<T...>::value;

template <isValueFunction... T>
requires is_same_value_function_check_v<T...>
struct get_first_value_function_type {
  using type = std::tuple_element_t<0, std::tuple<T...>>;
};

/// @brief Get a generic Value Function from the first element of the parameter pack
template <isValueFunction... T>
requires is_same_value_function_check_v<T...>
struct get_first_value_function_type_generic {

  using vFunctType = typename get_first_value_function_type<T...>::type;

  using type = ValueFunction<
      typename vFunctType::KeyMaker,
      typename vFunctType::ValueType,
      vFunctType::initial_value,
      vFunctType::discount_rate>;
};
} // namespace policy::objectives