#pragma once

#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "environment.hpp"
#include "policy/objectives/finite_value.hpp"
#include "policy/objectives/step_size.hpp"
#include "policy/objectives/value.hpp"
#include "policy/objectives/value_function.hpp"
#include "policy/objectives/value_function_keymaker.hpp"

namespace policy::objectives {

template <isValueFunctionKeymaker KEYMAPPER_T, isValue VALUE_T>
struct FiniteValueFunctionMapGetter {
  using KeyMaker = KEYMAPPER_T;
  using KeyType = typename KeyMaker::KeyType;
  using ValueType = VALUE_T;
  using Hash = typename KeyMaker::Hash;
  // When the key type is state. this is v(s) when the key type is (s,a) this
  // q(s, a) the q_table.
  using type = std::unordered_map<KeyType, ValueType, Hash>;
};

template <
    isValueFunction VALUE_FUNCTION_T,
    isStepSizeTaker INCREMENTAL_STEPSIZE_T = weighted_average_step_size_taker<typename VALUE_FUNCTION_T::ValueType>>
struct FiniteValueFunction
    : virtual VALUE_FUNCTION_T,
      virtual FiniteValueFunctionMapGetter<typename VALUE_FUNCTION_T::KeyMaker, typename VALUE_FUNCTION_T::ValueType>::
          type {

public:
  SETUP_TYPES_FROM_NESTED_ENVIRON(SINGLE_ARG(VALUE_FUNCTION_T::EnvironmentType));
  using ValueFunctionBaseType = VALUE_FUNCTION_T;
  using KeyMaker = typename VALUE_FUNCTION_T::KeyMaker;
  using KeyType = typename KeyMaker::KeyType;
  using ValueType = typename VALUE_FUNCTION_T::ValueType;
  using ValueTableType =
      typename FiniteValueFunctionMapGetter<typename VALUE_FUNCTION_T::KeyMaker, typename VALUE_FUNCTION_T::ValueType>::
          type;
  using StepSizeTaker = INCREMENTAL_STEPSIZE_T;

  constexpr static auto iterations = 1000;
  typename ValueType::Factory valueFactory{};

  FiniteValueFunction() = default;
  FiniteValueFunction(const FiniteValueFunction &) = default;

  void initialize(EnvironmentType &environment) override;

  /// @brief Extra getter to yield the value no matter the underlying
  /// type being held within the valueEstimates table. value is always a member.
  virtual PrecisionType valueAt(const KeyType &s);
  virtual PrecisionType valueAt(const EnvironmentType &e, const StateType &s, const ActionSpace &a);

  using VALUE_FUNCTION_T::operator();
  ValueType operator()(const KeyType &k) const override;
  ValueType operator()(const EnvironmentType &e, const StateType &s, const ActionSpace &a);

  void incrementalUpdate(const EnvironmentType &e, const TransitionType &s);
  void prettyPrint();
  KeyType getArgmaxKey(const EnvironmentType &e, const StateType &s) const override;
};

/// @brief Extra getter to yield the value no matter the underlying
/// type being held within the valueEstimates table. value is always a member.
template <isValueFunction V, isStepSizeTaker S>
auto FiniteValueFunction<V, S>::valueAt(const KeyType &s) -> PrecisionType {
  return this->emplace(s, valueFactory.create(this->initial_value, 1)).first->second.value;
}

template <isValueFunction V, isStepSizeTaker S>
auto FiniteValueFunction<V, S>::valueAt(const EnvironmentType &e, const StateType &s, const ActionSpace &a)
    -> PrecisionType {
  return this->valueAt(KeyMaker::make(e, s, a));
}

template <isValueFunction V, isStepSizeTaker S>
auto FiniteValueFunction<V, S>::initialize(EnvironmentType &environment) -> void {

  if constexpr (environment::FullyKnownConditionalStateActionEnvironment<EnvironmentType>) {
    /// Since we know the model well enough to ask the environment to generate
    /// all possible states we do so.
    for (const auto &state : environment.getAllPossibleStates()) {
      for (const auto &action : environment.getReachableActions(state)) {
        this->valueAt(KeyMaker::make(environment, state, action));
      }
    }

  } else if constexpr (environment::FullyKnownFiniteStateEnvironment<EnvironmentType>) {

    for (const auto &state : environment.getAllPossibleStates()) {
      for (const auto &action : environment.getAllPossibleActions()) {
        this->valueAt(KeyMaker::make(environment, state, action));
      }
    }

  } else {
    // TODO : USE RANDOM STATE GEN?
    // Random initialisation since we dont have a way to create states
  }
}

template <isValueFunction V, isStepSizeTaker S>
auto FiniteValueFunction<V, S>::operator()(const KeyType &k) const -> ValueType {
  return this->at(k);
}

template <isValueFunction V, isStepSizeTaker S>
auto FiniteValueFunction<V, S>::operator()(const EnvironmentType &e, const StateType &s, const ActionSpace &a)
    -> ValueType {
  return operator()(KeyMaker::make(e, s, a));
}

template <isValueFunction V, isStepSizeTaker S>
auto FiniteValueFunction<V, S>::incrementalUpdate(const EnvironmentType &e, const TransitionType &s) -> void {

  // Reward for this transition
  auto reward = RewardType::reward(s);
  auto key = KeyMaker::make(e, s.state, s.action);
  if (this->find(key) != this->end()) {
    auto &v = this->at(key);
    v.value = v.value + StepSizeTaker::getStepSize(v) * (reward - v.value);
    v.step++;
  } else {
    this->emplace(key, this->valueFactory.create(reward, 1));
  }

  // Go over all the other actions and update them with their global callback
  // A better mechanism might be to start with a state factory that holds
  // global state
  this->valueFactory.update();
}

template <isValueFunction V, isStepSizeTaker S>
auto FiniteValueFunction<V, S>::prettyPrint() -> void {
  for (const auto &[key, value] : *this) {
    std::cout << key << " : " << this->valueAt(key) << std::endl;
  }
}

template <isValueFunction V, isStepSizeTaker S>
auto FiniteValueFunction<V, S>::getArgmaxKey(const EnvironmentType &e, const StateType &s) const -> KeyType {
  auto availableActions = e.getReachableActions(s);
  auto maxIdx = std::max_element(this->begin(), this->end(), [&e, &availableActions](const auto &p1, const auto &p2) {
    if (availableActions.find(KeyMaker::get_action_from_key(e, p2.first)) != availableActions.end())
      return p1.second < p2.second;
    return false;
  });

  if (maxIdx == this->end())
    return KeyMaker::make(e, s, *availableActions.begin()); // or throw a runtime error here...

  return maxIdx->first;
}

#define SETUP_FINITE_VALUE_FUNCTION_TYPES(VALUE_FN_T, VALUE_T)                                                         \
  SETUP_VALUE_FUNCTION_TYPES(SINGLE_ARG(VALUE_FN_T));                                                                  \
  using ValueType = VALUE_T;

template <typename T>
concept isFiniteStateValueFunction = std::is_base_of_v<FiniteValueFunction<typename T::ValueFunctionBaseType>, T>;
template <typename T>
concept isFiniteValueFunction = std::is_base_of_v<FiniteValueFunction<typename T::ValueFunctionBaseType>, T>;

template <
    isValueFunctionKeymaker KEYMAPPER_T,
    isFiniteValue VALUE_T = Value<typename KEYMAPPER_T::EnvironmentType>,
    auto INITIAL_VALUE = 0.0F,
    auto DISCOUNT_RATE = 0.0F>
using FiniteValueFunctionHelper =
    FiniteValueFunction<ValueFunction<KEYMAPPER_T, VALUE_T, INITIAL_VALUE, DISCOUNT_RATE>>;

template <
    environment::FiniteEnvironmentType E,
    auto INITIAL_VALUE = 0.0F,
    auto DISCOUNT_RATE = 0.0F,
    template <typename> typename VALUE_C = FiniteValue>
// requires isFiniteValueTemplate<VALUE_C>
using FiniteStateActionValueFunction =
    FiniteValueFunction<StateActionValueFunction<E, INITIAL_VALUE, DISCOUNT_RATE, VALUE_C>>;

template <
    environment::FiniteEnvironmentType E,
    auto INITIAL_VALUE = 0.0F,
    auto DISCOUNT_RATE = 0.0F,
    template <typename> typename VALUE_C = FiniteValue>
requires isFiniteValueTemplate<VALUE_C>
using FiniteStateValueFunction = FiniteValueFunction<StateValueFunction<E, INITIAL_VALUE, DISCOUNT_RATE, VALUE_C>>;

template <
    environment::FiniteEnvironmentType E,
    auto INITIAL_VALUE = 0.0F,
    auto DISCOUNT_RATE = 0.0F,
    template <typename> typename VALUE_C = FiniteValue>
requires isFiniteValueTemplate<VALUE_C>
using FiniteActionValueFunction = FiniteValueFunction<ActionValueFunction<E, INITIAL_VALUE, DISCOUNT_RATE, VALUE_C>>;

} // namespace policy::objectives
