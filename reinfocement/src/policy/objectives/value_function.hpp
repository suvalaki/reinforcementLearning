#pragma once

#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "environment.hpp"
#include "policy/objectives/value.hpp"
#include "policy/objectives/value_function_keymaker.hpp"

#define VFT ValueFunction<KEYMAPPER_T, VALUE_T, INITIAL_VALUE, DISCOUNT_RATE>

namespace policy::objectives {

/** @brief This is a generic value function which can be specialised to map
 * (state, aciton) -> keys and keys -> values.
 *
 * @tparam KEYMAPPER_T turns (state, action) pairs into a single value which is
 * the domain for the value function. It could be the case that the number of
 * keys is much smaller than the number of state-action permutations.
 */
template <isValueFunctionKeymaker KEYMAPPER_T,
          isValue VALUE_T = Value<typename KEYMAPPER_T::EnvironmentType>,
          auto INITIAL_VALUE = 0.0F,
          auto DISCOUNT_RATE = 0.0F>
requires std::is_same_v<typename KEYMAPPER_T::EnvironmentType, typename VALUE_T::EnvironmentType>
struct ValueFunction {

  SETUP_TYPES_FROM_NESTED_ENVIRON(SINGLE_ARG(KEYMAPPER_T::EnvironmentType));
  using KeyMaker = KEYMAPPER_T;
  using KeyType = typename KeyMaker::KeyType;
  using ValueType = VALUE_T;
  using ValueFactory = typename ValueType::Factory;

  ValueFunction(const PrecisionType &initial_value = INITIAL_VALUE,
                const PrecisionType &discount_rate = DISCOUNT_RATE) {}
  ValueFunction(const ValueFunction &) = default;

  // The starting value estimate
  constexpr static PrecisionType initial_value = INITIAL_VALUE;
  constexpr static PrecisionType discount_rate = DISCOUNT_RATE;

  virtual void initialize(EnvironmentType &environment) = 0;
  virtual ValueType operator()(const KeyType &k) const = 0;

  static KeyType makeKey(const EnvironmentType &environment, const StateType &s, const ActionSpace &a);
  PrecisionType &valueAt(const KeyType &s) const;
  virtual KeyType getArgmaxKey(const EnvironmentType &e, const StateType &s) const = 0;
};

template <typename T>
concept isValueFunction =
    std::is_base_of_v<ValueFunction<typename T::KeyMaker, typename T::ValueType, T::initial_value, T::discount_rate>,
                      T>;

template <isValueFunctionKeymaker KEYMAPPER_T, isValue VALUE_T, auto INITIAL_VALUE, auto DISCOUNT_RATE>
typename VFT::KeyType VFT::makeKey(const EnvironmentType &e, const StateType &s, const ActionSpace &a) {
  return KeyMaker::make(e, s, a);
}

template <isValueFunctionKeymaker KEYMAPPER_T, isValue VALUE_T, auto INITIAL_VALUE, auto DISCOUNT_RATE>
typename VFT::PrecisionType &VFT::valueAt(const KeyType &k) const {
  return this->operator()(k).getValue();
}

/** @brief A mapping fromm (state, action) to value. q(s, a). Estimating this
 * is required when the transition model isnt present (as is the case for
 * monte carlo model free approximation).
 */
template <environment::EnvironmentType E,
          auto INITIAL_VALUE = 0.0F,
          auto DISCOUNT_RATE = 0.0F,
          template <typename> typename VALUE_T = Value>
requires isValueTemplate<VALUE_T>
using StateActionValueFunction = ValueFunction<StateActionKeymaker<E>, VALUE_T<E>, INITIAL_VALUE, DISCOUNT_RATE>;

template <typename T>
concept isStateActionValueFunction =
    isValueFunction<T> && std::is_same_v<typename T::KeyMaker, StateActionKeymaker<typename T::EnvironmentType>>;

/** @brief A mapping of states to values v(s). When the finite model is fully
 * known this is enough to calculcate future expected returns under a given
 * policy ( as is the case in MDP).
 */
template <environment::EnvironmentType E,
          auto INITIAL_VALUE = 0.0F,
          auto DISCOUNT_RATE = 0.0F,
          template <typename> typename VALUE_T = Value>
requires isValueTemplate<VALUE_T>
using StateValueFunction = ValueFunction<StateKeymaker<E>, VALUE_T<E>, INITIAL_VALUE, DISCOUNT_RATE>;

template <typename T>
concept isStateValueFunction =
    isValueFunction<T> && std::is_same_v<typename T::KeyMaker, StateValueFunction<typename T::EnvironmentType>>;

/** @brief A mapping of actions to valus : a -> q(a)
 */
template <environment::EnvironmentType E,
          auto INITIAL_VALUE = 0.0F,
          auto DISCOUNT_RATE = 0.0F,
          template <typename> typename VALUE_T = Value>
requires isValueTemplate<VALUE_T>
using ActionValueFunction = ValueFunction<ActionKeymaker<E>, VALUE_T<E>, INITIAL_VALUE, DISCOUNT_RATE>;

template <typename T>
concept isActionValueFunction =
    isValueFunction<T> && std::is_same_v<typename T::KeyMaker, ActionValueFunction<typename T::EnvironmentType>>;

} // namespace policy::objectives

#undef VFT