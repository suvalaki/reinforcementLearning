#pragma once
#include <xtensor/xfixed.hpp>
#include <xtensor/xrandom.hpp>

#include "action.hpp"
#include "dummy_environment.hpp"
#include "environment.hpp"
#include "spec.hpp"

namespace policy::objectives {

template <typename T>
concept isValueFunctionKeymaker = requires(T t) {
  typename T::KeyType;
  typename T::Hash;
  {
    T::make(std::declval<typename T::EnvironmentType>(),
            std::declval<typename T::StateType>(),
            std::declval<typename T::ActionSpace>())
    } -> std::same_as<typename T::KeyType>;
  {
    T::get_action_from_key(std::declval<typename T::EnvironmentType>(), std::declval<typename T::KeyType>())
    } -> std::same_as<typename T::ActionSpace>;
  { T::hash(std::declval<typename T::KeyType>()) } -> std::same_as<std::size_t>;
};

template <typename T> struct HashBuilder {
  std::size_t operator()(const typename T::KeyType &key) const { return T::hash(key); }
};

// Interface
template <environment::EnvironmentType ENVIRON_T, typename KEY_T> struct ValueFunctionKeymaker {

  SETUP_TYPES_FROM_ENVIRON(SINGLE_ARG(ENVIRON_T));

  // The type that will be used for Q value lookup
  using KeyType = KEY_T;
  using Hash = HashBuilder<ValueFunctionKeymaker<ENVIRON_T, KEY_T>>;

  static KeyType make(const EnvironmentType &e, const StateType &s, const ActionSpace &action);
  static StateType get_state_from_key(const EnvironmentType &e, const KeyType &key);
  static ActionSpace get_action_from_key(const EnvironmentType &e, const KeyType &key);
  static std::size_t hash(const KEY_T &key);
};

template <template <typename, typename> class C>
concept isValueFunctionKeymakerTemplate =
    std::is_base_of_v<ValueFunctionKeymaker<environment::dummy::DummyEnvironment, void>,
                      C<environment::dummy::DummyEnvironment, void>>;

#define SETUP_KEYMAKER_TYPES(BASE_T)                                                                                   \
  SETUP_TYPES(SINGLE_ARG(BASE_T))                                                                                      \
  using EnvironmentType = typename BASE_T::EnvironmentType;                                                            \
  using KeyType = typename BASE_T::KeyType;

template <environment::EnvironmentType ENVIRON_T>
struct StateActionKeymaker
    : ValueFunctionKeymaker<ENVIRON_T, std::pair<typename ENVIRON_T::StateType, typename ENVIRON_T::ActionSpace>> {

  SETUP_KEYMAKER_TYPES(SINGLE_ARG(
      ValueFunctionKeymaker<ENVIRON_T, std::pair<typename ENVIRON_T::StateType, typename ENVIRON_T::ActionSpace>>))

  static KeyType make(const EnvironmentType &e, const StateType &s, const ActionSpace &action) {
    return std::make_pair(s, action);
  }
  static StateType get_state_from_key(const EnvironmentType &e, const KeyType &key) { return key.first; }
  static ActionSpace get_action_from_key(const EnvironmentType &e, const KeyType &key) { return key.second; }
  static std::size_t hash(const KeyType &key) { return key.first.hash() ^ key.second.hash(); }

  struct Hash {
    std::size_t operator()(const KeyType &key) const { return StateActionKeymaker::hash(key); }
  };
};

template <typename T>
concept isStateActionKeymaker = std::is_base_of_v<StateActionKeymaker<typename T::EnvironmentType>, T>;

template <environment::EnvironmentType ENVIRON_T>
struct ActionKeymaker : ValueFunctionKeymaker<ENVIRON_T, typename ENVIRON_T::ActionSpace> {

  SETUP_KEYMAKER_TYPES(SINGLE_ARG(ValueFunctionKeymaker<ENVIRON_T, typename ENVIRON_T::ActionSpace>))
  // using Hash = HashBuilder<ActionKeymaker<ENVIRON_T>>;

  static KeyType make(const EnvironmentType &e, const StateType &s, const ActionSpace &action) { return action; }
  static StateType get_state_from_key(const EnvironmentType &e, const KeyType &key) { return e.getNullState(); }
  static ActionSpace get_action_from_key(const EnvironmentType &e, const KeyType &key) { return key; }
  static std::size_t hash(const KeyType &key) { return key.hash(); }

  struct Hash {
    std::size_t operator()(const KeyType &key) const { return ActionKeymaker::hash(key); }
  };
};

template <typename T>
concept isActionKeymaker = std::is_base_of_v<ActionKeymaker<typename T::EnvironmentType>, T>;

template <environment::EnvironmentType ENVIRON_T>
struct StateKeymaker : ValueFunctionKeymaker<ENVIRON_T, typename ENVIRON_T::StateType> {

  SETUP_KEYMAKER_TYPES(SINGLE_ARG(ValueFunctionKeymaker<ENVIRON_T, typename ENVIRON_T::StateType>))
  using Hash = HashBuilder<StateKeymaker<ENVIRON_T>>;

  static KeyType make(const EnvironmentType &e, const StateType &s, const ActionSpace &action) { return s; }
  static StateType get_state_from_key(const EnvironmentType &e, const KeyType &key) { return key; }
  static ActionSpace get_action_from_key(const EnvironmentType &e, const KeyType &key) { return ActionSpace(); }
  static std::size_t hash(const KeyType &key) { return key.hash(); }
};

// Helper to print the default key type
template <typename T0, typename T1> std::ostream &operator<<(std::ostream &os, const std::pair<T0, T1> &p) {
  os << "(" << p.first << ", " << p.second << ")";
  return os;
}

template <typename T>
concept isStateKeymaker = std::is_base_of_v<StateKeymaker<typename T::EnvironmentType>, T>;

} // namespace policy::objectives