#pragma once

#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "environment.hpp"
#include "policy/state_action_keymaker.hpp"

namespace policy {

/** @brief This is a generic value function which can be specialised to map
 * (state, aciton) -> keys and keys -> values.
 *
 * @tparam KEYMAPPER_T turns (state, action) pairs into a single value which is
 * the domain for the value function. It could be the case that the number of
 * keys is much smaller than the number of state-action permutations.
 */
template <environment::EnvironmentType ENVIRONMENT_T,
          isStateActionKeymaker KEYMAPPER_T = DefaultActionKeymaker<ENVIRONMENT_T>,
          auto INITIAL_VALUE = 0.0F,
          auto DISCOUNT_RATE = 0.0F>
struct ValueFunctionPrototype {

  SETUP_TYPES(SINGLE_ARG(ENVIRONMENT_T));
  using EnvironmentType = ENVIRONMENT_T;
  using KeyMaker = KEYMAPPER_T;
  using KeyType = typename KeyMaker::KeyType;

  // The starting value estimate
  constexpr static PrecisionType initial_value = INITIAL_VALUE;
  constexpr static PrecisionType discount_rate = DISCOUNT_RATE;

  virtual PrecisionType valueAt(const KeyType &s) = 0;
  virtual void initialize(EnvironmentType &environment) = 0;
};

template <typename T>
concept isValueFunctionPrototype = std::is_base_of_v<
    ValueFunctionPrototype<typename T::EnvironmentType, typename T::KeyMaker, T::initial_value, T::discount_rate>,
    T>;

template <isStateActionKeymaker KEYMAPPER_T,
          isStateActionValue VALUE_T = StateActionValue<typename KEYMAPPER_T::EnvironmentType>>
struct FiniteValueFunctionMapGetter {
  using KeyMaker = KEYMAPPER_T;
  using KeyType = typename KeyMaker::KeyType;
  using ValueType = VALUE_T;
  using QTableValueType = ValueType;
  using Hash = typename KeyMaker::Hash;
  // When the key type is state. this is v(s) when the key type is (s,a) this
  // q(s, a) the q_table.
  using type = std::unordered_map<KeyType, QTableValueType, Hash>;
};

template <isValueFunctionPrototype VALUE_FUNCTION_T, isStateActionValue VALUE_T>
struct FiniteValueFunctionMixin
    : public FiniteValueFunctionMapGetter<typename VALUE_FUNCTION_T::KeyMaker, VALUE_T>::type,
      VALUE_FUNCTION_T {
  using ValueFunctionBaseType = VALUE_FUNCTION_T;
  using EnvironmentType = typename VALUE_FUNCTION_T::EnvironmentType;
  using PrecisionType = typename EnvironmentType::PrecisionType;
  using StateType = typename EnvironmentType::StateType;
  using KeyMaker = typename VALUE_FUNCTION_T::KeyMaker;
  using KeyType = typename KeyMaker::KeyType;
  using ValueType = VALUE_T;
  using QTableValueType = ValueType;

  constexpr static typename EnvironmentType::PrecisionType initial_value = VALUE_FUNCTION_T::initial_value;

  constexpr static auto iterations = 1000;

  typename ValueType::Factory valueFactory{};

  /// @brief Extra getter to yield the value no matter the underlying
  /// type being held within the valueEstimates table. value is always a member.
  PrecisionType valueAt(const KeyType &s) {
    return this->emplace(s, valueFactory.create(initial_value, 1)).first->second.value;
  }

  void initialize(EnvironmentType &environment) {

    if constexpr (environment::FullyKnownConditionalStateActionEnvironment<EnvironmentType>) {
      /// Since we know the model well enough to ask the environment to generate
      /// all possible states we do so.
      for (const auto &state : environment.getAllPossibleStates()) {
        for (const auto &action : environment.getReachableActions(state)) {
          this->valueAt(KeyMaker::make(state, action));
        }
      }

    } else if constexpr (environment::FullyKnownFiniteStateEnvironment<EnvironmentType>) {

      // For every state which we know lets try some random acitons
      for (const auto &state : environment.getAllPossibleStates()) {
        auto randomPolicy = policy::RandomPolicy<EnvironmentType>();
        environment.state = state;
        for (int i = 0; i < iterations; i++) {
          auto recommendedAction = randomPolicy(environment.state);
          auto transition = environment.step(recommendedAction);
          this->valueAt(KeyMaker::make(transition.state, transition.action));
        }
      }

    } else {
      // TODO : USE RANDOM STATE GEN?
      // Random initialisation since we dont have a way to create states
      auto randomPolicy = policy::RandomPolicy<EnvironmentType>();
      for (int i = 0; i < iterations; i++) {
        auto recommendedAction = randomPolicy(environment.state);
        auto transition = environment.step(recommendedAction);
        this->valueAt(KeyMaker::make(transition.state, transition.action));
      }
    }
  }

  void prettyPrint() const {
    for (const auto &[key, value] : *this) {
      std::cout << key << " : " << value.value << std::endl;
    }
  }
};

template <typename T>
concept isFiniteStateValueFunction =
    std::is_base_of_v<FiniteValueFunctionMixin<typename T::ValueFunctionBaseType, typename T::ValueType>, T>;

/** @brief A mapping fromm (state, action) to value. q(s, a). Estimating this
 * is required when the transition model isnt present (as is the case for
 * monte carlo model free approximation).
 */
template <environment::EnvironmentType ENVIRONMENT_T, auto INITIAL_VALUE = 0.0F, auto DISCOUNT_RATE = 0.0F>
using StateActionValueFunction =
    ValueFunctionPrototype<ENVIRONMENT_T, DefaultActionKeymaker<ENVIRONMENT_T>, INITIAL_VALUE, DISCOUNT_RATE>;

/** @brief A mapping of states to values v(s). When the finite model is fully
 * known this is enough to calculcate future expected returns under a given
 * policy ( as is the case in MDP).
 */
template <environment::EnvironmentType ENVIRONMENT_T, auto INITIAL_VALUE = 0.0F, auto DISCOUNT_RATE = 0.0F>
using StateValueFunction =
    ValueFunctionPrototype<ENVIRONMENT_T, StateKeymaker<ENVIRONMENT_T>, INITIAL_VALUE, DISCOUNT_RATE>;

/** @brief A mapping of actions to valus : a -> q(a)
 */
template <environment::EnvironmentType ENVIRONMENT_T, auto INITIAL_VALUE = 0.0F, auto DISCOUNT_RATE = 0.0F>
using ActionValueFunction =
    ValueFunctionPrototype<ENVIRONMENT_T, ActionKeymaker<ENVIRONMENT_T>, INITIAL_VALUE, DISCOUNT_RATE>;

template <environment::EnvironmentType ENVIRONMENT_T,
          isStateActionValue VALUE_T = StateActionValue<ENVIRONMENT_T>,
          auto INITIAL_VALUE = 0.0F,
          auto DISCOUNT_RATE = 0.0F>
struct FiniteStateActionValueFunction
    : FiniteValueFunctionMixin<StateActionValueFunction<ENVIRONMENT_T, INITIAL_VALUE, DISCOUNT_RATE>, VALUE_T> {};

template <environment::EnvironmentType ENVIRONMENT_T,
          isStateActionValue VALUE_T = StateActionValue<ENVIRONMENT_T>,
          auto INITIAL_VALUE = 0.0F,
          auto DISCOUNT_RATE = 0.0F>
struct FiniteStateValueFunction
    : FiniteValueFunctionMixin<StateValueFunction<ENVIRONMENT_T, INITIAL_VALUE, DISCOUNT_RATE>, VALUE_T> {};

template <environment::EnvironmentType ENVIRONMENT_T,
          isStateActionValue VALUE_T = StateActionValue<ENVIRONMENT_T>,
          auto INITIAL_VALUE = 0.0F,
          auto DISCOUNT_RATE = 0.0F>
struct FiniteActionValueFunction
    : FiniteValueFunctionMixin<ActionValueFunction<ENVIRONMENT_T, INITIAL_VALUE, DISCOUNT_RATE>, VALUE_T> {};

} // namespace policy
