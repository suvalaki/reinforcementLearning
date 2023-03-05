#pragma once
#include <type_traits>

#include "policy/objectives/finite_value_function.hpp"
#include "policy/objectives/value_function_combination.hpp"

namespace policy::objectives {

template <isFiniteValueFunction... VALUE_FUNCTION_T> struct is_finite_value_function_check;
template <isFiniteValueFunction VALUE_FUNCTION_T>
struct is_finite_value_function_check<VALUE_FUNCTION_T> : std::true_type {};
template <isFiniteValueFunction VALUE_FUNCTION_T,
          isFiniteValueFunction VALUE_FUNCTION_T2,
          isFiniteValueFunction... VALUE_FUNCTION_Ts>
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
  // use a generic finite value function that we can override the virtual methods of
  using type = FiniteValueFunction<typename vFunctType::ValueFunctionBaseType, typename vFunctType::StepSizeTaker>;
};

template <isFiniteValueFunction... VALUE_FUNCTION_T>
struct AdditiveFiniteValueFunctionCombination
    : AdditiveValueFunctionCombination<VALUE_FUNCTION_T...>,
      get_first_finite_value_function_type_generic<VALUE_FUNCTION_T...>::type {

  using BaseType = AdditiveValueFunctionCombination<VALUE_FUNCTION_T...>;
  AdditiveFiniteValueFunctionCombination(auto &&...args)
      : AdditiveValueFunctionCombination<VALUE_FUNCTION_T...>(args...) {}
  AdditiveFiniteValueFunctionCombination(const AdditiveFiniteValueFunctionCombination &p)
      : AdditiveValueFunctionCombination<VALUE_FUNCTION_T...>(p) {}
  AdditiveFiniteValueFunctionCombination() = delete;

  using fValueFunctionType = get_first_finite_value_function_type_generic<VALUE_FUNCTION_T...>::type;
  using ValueType = typename fValueFunctionType::ValueType;
  using KeyMaker = typename fValueFunctionType::KeyMaker;
  using KeyType = typename fValueFunctionType::KeyType;
  using ValueTableType = typename fValueFunctionType::ValueTableType;

  SETUP_TYPES_FROM_NESTED_ENVIRON(SINGLE_ARG(BaseType::EnvironmentType));

  using BaseType::initialize;

  // Override the FiniteValueFunction virtual members to give the correct summation instead
  ValueType operator()(const KeyType &k) const override { return BaseType::operator()(k); }
  PrecisionType valueAt(const KeyType &k) override { return BaseType::operator()(k).value; }

  // We intercept the call to the value function and return the sum of the values returned by each value function.

  KeyType getArgmaxKey(const EnvironmentType &e, const StateType &s) const override {

    // Construct an in memory additive value function by incrementally adding each value function
    // to the previous one.
    // Fill up the possible keys as a combination of the keys of each value function
    auto tmpValueFunction = ValueTableType();
    const auto inserter = [&tmpValueFunction](const auto &k) -> void { tmpValueFunction[k.first] += k.second; };
    std::apply([&tmpValueFunction, &inserter](
                   const auto &...vFuncts) { (..., (std::for_each(vFuncts.begin(), vFuncts.end(), inserter))); },
               this->valueFunctions);

    auto availableActions = e.getReachableActions(s);
    auto maxIdx = std::max_element(
        tmpValueFunction.begin(), tmpValueFunction.end(), [&e, &availableActions](const auto &p1, const auto &p2) {
          if (availableActions.find(KeyMaker::get_action_from_key(e, p2.first)) != availableActions.end())
            return p1.second < p2.second;
          return false;
        });

    if (maxIdx == tmpValueFunction.end())
      return KeyMaker::make(e, s, *availableActions.begin()); // or throw a runtime error here...

    return maxIdx->first;
  }
};

template <typename... T> struct getter_AdditiveFiniteValueFunctionCombination;
template <typename... T> struct getter_AdditiveFiniteValueFunctionCombination<std::tuple<T...>> {
  using type = AdditiveFiniteValueFunctionCombination<T...>;
};

template <typename T>
concept isFiniteAdditiveValueFunctionCombination =
    std::is_base_of_v<typename getter_AdditiveFiniteValueFunctionCombination<typename T::ValueFunctionTypes>::type, T>;

} // namespace policy::objectives
