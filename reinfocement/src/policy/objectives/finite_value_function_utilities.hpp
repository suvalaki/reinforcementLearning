#pragma once
#include <type_traits>

#include "policy/objectives/finite_value_function.hpp"
#include "policy/objectives/value_function_utilities.hpp"

namespace policy::objectives {

/// @brief Get a generic Finite Value Function from the first element of the parameter pack of value functions where all
/// of the value functions share the same ValueType to output
template <isFiniteValueFunction... V>
requires is_same_value_function_check_v<V...>
struct get_first_finite_value_function_type_generic {
  //
  using vFunctType = typename get_first_value_function_type<V...>::type;
  // use a generic finite value function that we can override the virtual methods of
  using type = FiniteValueFunction<typename vFunctType::ValueFunctionBaseType, typename vFunctType::StepSizeTaker>;
};

} // namespace policy::objectives