#pragma once
#include <tuple>
#include <type_traits>

#include "reinforce/policy/objectives/value_function.hpp"
#include "reinforce/policy/objectives/value_function_utilities.hpp"

namespace policy::objectives {

/// @brief A base template interface for use in defining different types of combinations of policies.
template <isValueFunction... V>
requires is_same_value_function_check_v<V...>
struct ValueFunctionCombination : get_first_value_function_type_generic<V...>::type {

  using ValueFunctionTypes = std::tuple<V...>;
  SETUP_TYPES_FROM_NESTED_ENVIRON(SINGLE_ARG(std::tuple_element_t<0, ValueFunctionTypes>::EnvironmentType));
  using ValueType = std::tuple_element_t<0, ValueFunctionTypes>::ValueType;
  using KeyType = std::tuple_element_t<0, ValueFunctionTypes>::KeyType;

  using ValueFunctionRefs = std::tuple<V &...>;
  ValueFunctionRefs valueFunctions;

  ValueFunctionCombination(V &...valueFunctions) : valueFunctions(valueFunctions...) {}
  ValueFunctionCombination(const ValueFunctionCombination &v) : valueFunctions(v.valueFunctions) {}

  void initialize(EnvironmentType &environment) override {
    std::apply([&](auto &...valueFunctions) { (valueFunctions.initialize(environment), ...); }, this->valueFunctions);
  }

  // How the combination is applied is an implementation detail for the specific combination.
};

/// @brief A combination of value functions that returns the sum of the values returned by each value function.
/// @tparam ...VALUE_FUNCTION_T
template <isValueFunction... V>
struct AdditiveValueFunctionCombination : ValueFunctionCombination<V...> {

  using BaseType = ValueFunctionCombination<V...>;
  AdditiveValueFunctionCombination(auto &&...args) : ValueFunctionCombination<V...>(args...) {}
  AdditiveValueFunctionCombination(const AdditiveValueFunctionCombination &v) : ValueFunctionCombination<V...>(v) {}

  SETUP_TYPES_FROM_NESTED_ENVIRON(SINGLE_ARG(BaseType::EnvironmentType));
  using ValueType = typename BaseType::ValueType;
  using KeyType = typename BaseType::KeyType;

  ValueType operator()(const KeyType &k) const override;
  // PrecisionType operator()(const KeyType &k, const ValueType &v) const override;
};

template <isValueFunction... V>
auto AdditiveValueFunctionCombination<V...>::operator()(const KeyType &k) const ->
    typename AdditiveValueFunctionCombination<V...>::ValueType {
  ValueType result = {};
  std::apply([&](auto &...valueFunctions) { ((result += valueFunctions[k]), ...); }, this->valueFunctions);
  return result;
}

} // namespace policy::objectives
