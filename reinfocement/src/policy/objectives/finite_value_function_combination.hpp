#pragma once
#include <type_traits>

#include "finite_value_function_utilities.hpp"
#include "policy/objectives/finite_value_function.hpp"
#include "policy/objectives/value_function_combination.hpp"
#include <ranges>

namespace policy::objectives {

template <isFiniteValueFunction... V>
struct AdditiveFiniteValueFunctionCombination : AdditiveValueFunctionCombination<V...>,
                                                get_first_finite_value_function_type_generic<V...>::type {

  using BaseType = AdditiveValueFunctionCombination<V...>;
  SETUP_TYPES_FROM_NESTED_ENVIRON(SINGLE_ARG(BaseType::EnvironmentType));
  using fValueFunctionType = get_first_finite_value_function_type_generic<V...>::type;
  using ValueType = typename fValueFunctionType::ValueType;
  using KeyMaker = typename fValueFunctionType::KeyMaker;
  using KeyType = typename fValueFunctionType::KeyType;
  using ValueTableType = typename fValueFunctionType::ValueTableType;

  AdditiveFiniteValueFunctionCombination(auto &&...args);
  AdditiveFiniteValueFunctionCombination(const AdditiveFiniteValueFunctionCombination &p);
  AdditiveFiniteValueFunctionCombination() = delete;

  using BaseType::initialize;
  ValueType operator()(const KeyType &k) const override;
  PrecisionType valueAt(const KeyType &k) override;
  KeyType getArgmaxKey(const EnvironmentType &e, const StateType &s) const override;

  ValueTableType createTemporaryValueTable() const;
  KeyType getMaxKeyFromTable(const EnvironmentType &e, const StateType &s, const ValueTableType &v) const;
};

template <isFiniteValueFunction... T>
AdditiveFiniteValueFunctionCombination<T...>::AdditiveFiniteValueFunctionCombination(auto &&...args)
    : AdditiveValueFunctionCombination<T...>(args...) {}

template <isFiniteValueFunction... T>
AdditiveFiniteValueFunctionCombination<T...>::AdditiveFiniteValueFunctionCombination(
    const AdditiveFiniteValueFunctionCombination &p)
    : AdditiveValueFunctionCombination<T...>(p) {}

template <isFiniteValueFunction... T>
auto AdditiveFiniteValueFunctionCombination<T...>::operator()(const KeyType &k) const -> ValueType {
  return BaseType::operator()(k);
}
template <isFiniteValueFunction... T>
auto AdditiveFiniteValueFunctionCombination<T...>::valueAt(const KeyType &k) -> PrecisionType {
  return BaseType::operator()(k).value;
}

template <isFiniteValueFunction... T>
auto AdditiveFiniteValueFunctionCombination<T...>::getArgmaxKey(const EnvironmentType &e, const StateType &s) const
    -> KeyType {

  const auto tmpValueFunction = this->createTemporaryValueTable();
  auto filteredTable = std::views::filter(tmpValueFunction, [&e, &s](const auto &p) {
    return e.isReachableAction(s, KeyMaker::get_action_from_key(e, p.first));
  });
  const auto key = this->getMaxKeyFromTable(e, s, tmpValueFunction);
  return key;
}

/** @brief Construct an additive construction of all value functions in this combination
 * @details Construct an in memory additive value function by incrementally adding each value function
 *  to the previous one. Then fill up the possible keys as a combination of the keys of each value function.
 *  This gives us the value of each possible action in the state.
 */
template <isFiniteValueFunction... T>
auto AdditiveFiniteValueFunctionCombination<T...>::createTemporaryValueTable() const -> ValueTableType {

  auto tmpValueFunction = ValueTableType();
  const auto inserter = [&tmpValueFunction](const auto &k) -> void { tmpValueFunction[k.first] += k.second; };
  std::apply(
      [&tmpValueFunction, &inserter](const auto &...vFuncts) {
        (..., (std::for_each(vFuncts.begin(), vFuncts.end(), inserter)));
      },
      this->valueFunctions);
  return tmpValueFunction;
}

/** @brief For the provided valueTable find the maximum key */
template <isFiniteValueFunction... T>
auto AdditiveFiniteValueFunctionCombination<T...>::getMaxKeyFromTable(
    const EnvironmentType &e, const StateType &s, const ValueTableType &tmpValueFunction) const -> KeyType {

  auto maxIdx = std::max_element(tmpValueFunction.begin(), tmpValueFunction.end(), [](const auto &p1, const auto &p2) {
    return p1.second < p2.second;
  });

  if (maxIdx == tmpValueFunction.end())
    return KeyMaker::make(e, s, *e.getReachableActions(s).begin()); // or throw a runtime error here...

  return maxIdx->first;
}

template <typename... T>
struct getter_AdditiveFiniteValueFunctionCombination;
template <typename... T>
struct getter_AdditiveFiniteValueFunctionCombination<std::tuple<T...>> {
  using type = AdditiveFiniteValueFunctionCombination<T...>;
};

template <typename T>
concept isFiniteAdditiveValueFunctionCombination =
    std::is_base_of_v<typename getter_AdditiveFiniteValueFunctionCombination<typename T::ValueFunctionTypes>::type, T>;

} // namespace policy::objectives
