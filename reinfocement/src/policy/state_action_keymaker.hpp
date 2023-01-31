#pragma once
#include <xtensor/xfixed.hpp>
#include <xtensor/xrandom.hpp>

#include "action.hpp"
#include "environment.hpp"
#include "spec.hpp"

namespace policy {

template <typename T> struct HashBuilder {
  std::size_t operator()(const typename T::KeyType &key) const { return T::hash(key); }
};

template <environment::EnvironmentType ENVIRON_T, typename KEY_T> struct StateActionKeymaker {

  SETUP_TYPES_FROM_ENVIRON(SINGLE_ARG(ENVIRON_T));

  // The type that will be used for Q value lookup
  using KeyType = KEY_T;

  static KeyType make(const StateType &s, const ActionSpace &action);
  static StateType get_state_from_key(const EnvironmentType &e, const KeyType &key);
  static ActionSpace get_action_from_key(const KeyType &key);
  static std::size_t hash(const KEY_T &key);

  using Hash = HashBuilder<StateActionKeymaker<ENVIRON_T, KEY_T>>;
};

template <typename T>
concept isStateActionKeymaker = requires(T t) {
  typename T::KeyType;
  {
    T::make(std::declval<typename T::StateType>(), std::declval<typename T::ActionSpace>())
    } -> std::same_as<typename T::KeyType>;
  { T::get_action_from_key(std::declval<typename T::KeyType>()) } -> std::same_as<typename T::ActionSpace>;
  { T::hash(std::declval<typename T::KeyType>()) } -> std::same_as<std::size_t>;
};

#define SETUP_KEYMAKER_TYPES(BASE_T)                                                                                   \
  SETUP_TYPES(SINGLE_ARG(BASE_T))                                                                                      \
  using EnvironmentType = typename BASE_T::EnvironmentType;                                                            \
  using KeyType = typename BASE_T::KeyType;

template <environment::EnvironmentType ENVIRON_T>
struct DefaultActionKeymaker
    : StateActionKeymaker<ENVIRON_T, std::pair<typename ENVIRON_T::StateType, typename ENVIRON_T::ActionSpace>> {

  SETUP_KEYMAKER_TYPES(SINGLE_ARG(
      StateActionKeymaker<ENVIRON_T, std::pair<typename ENVIRON_T::StateType, typename ENVIRON_T::ActionSpace>>))

  static KeyType make(const StateType &s, const ActionSpace &action) { return std::make_pair(s, action); }

  static typename ENVIRON_T::StateType
  get_state_from_key(const EnvironmentType &e,
                     const std::pair<typename ENVIRON_T::StateType, typename ENVIRON_T::ActionSpace> &key) {
    return key.first;
  }

  static typename ENVIRON_T::ActionSpace
  get_action_from_key(const std::pair<typename ENVIRON_T::StateType, typename ENVIRON_T::ActionSpace> &key) {
    return key.second;
  }

  static std::size_t hash(const std::pair<typename ENVIRON_T::StateType, typename ENVIRON_T::ActionSpace> &key) {
    return key.first.hash() ^ key.second.hash();
  }

  using Hash = HashBuilder<DefaultActionKeymaker<ENVIRON_T>>;
};

template <environment::EnvironmentType ENVIRON_T>
struct ActionKeymaker : StateActionKeymaker<ENVIRON_T, typename ENVIRON_T::ActionSpace> {

  SETUP_KEYMAKER_TYPES(SINGLE_ARG(StateActionKeymaker<ENVIRON_T, typename ENVIRON_T::ActionSpace>))

  static KeyType make(const StateType &s, const ActionSpace &action) { return action; }

  static typename ENVIRON_T::StateType get_state_from_key(const EnvironmentType &e, const KeyType &key) {
    // Always return the null type
    return e.getNullState();
  }

  static typename ENVIRON_T::ActionSpace get_action_from_key(const KeyType &key) { return key; }

  static std::size_t hash(const KeyType &key) { return key.hash(); }

  using Hash = HashBuilder<ActionKeymaker<ENVIRON_T>>;
};

template <environment::EnvironmentType ENVIRON_T>
struct StateKeymaker : StateActionKeymaker<ENVIRON_T, typename ENVIRON_T::StateType> {

  SETUP_KEYMAKER_TYPES(SINGLE_ARG(StateActionKeymaker<ENVIRON_T, typename ENVIRON_T::StateType>))

  static KeyType make(const StateType &s, const ActionSpace &action) { return s; }

  static typename ENVIRON_T::StateType get_state_from_key(const EnvironmentType &e, const KeyType &key) {
    // Always return the null type
    return key;
  }

  static typename ENVIRON_T::ActionSpace get_action_from_key(const KeyType &key) { return ActionSpace(); }

  static std::size_t hash(const KeyType &key) { return key.hash(); }

  using Hash = HashBuilder<StateKeymaker<ENVIRON_T>>;
};

// Helper to print the default key type
template <typename T0, typename T1> std::ostream &operator<<(std::ostream &os, const std::pair<T0, T1> &p) {
  os << "(" << p.first << ", " << p.second << ")";
  return os;
}
} // namespace policy