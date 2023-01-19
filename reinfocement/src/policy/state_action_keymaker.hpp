#pragma once
#include <xtensor/xfixed.hpp>
#include <xtensor/xrandom.hpp>

#include "action.hpp"
#include "environment.hpp"
#include "spec.hpp"

namespace policy {

template <typename T> struct HashBuilder {
  std::size_t operator()(const typename T::KeyType &key) const {
    return T::hash(key);
  }
};

template <environment::EnvironmentType ENVIRON_T, typename KEY_T>
struct StateActionKeymaker {
  using EnvironmentType = typename ENVIRON_T::EnvironmentType;
  using StateType = typename EnvironmentType::StateType;
  using ActionSpace = typename EnvironmentType::ActionSpace;

  // The type that will be used for Q value lookup
  using KeyType = KEY_T;

  static KeyType make(const StateType &s, const ActionSpace &action);
  static ActionSpace get_action_from_key(const KeyType &key);
  static std::size_t hash(const KEY_T &key);

  using Hash = HashBuilder<StateActionKeymaker<ENVIRON_T, KEY_T>>;
};

template <typename T>
concept isStateActionKeymaker = requires(T t) {
  typename T::KeyType;
  {
    T::make(std::declval<typename T::StateType>(),
            std::declval<typename T::ActionSpace>())
    } -> std::same_as<typename T::KeyType>;
  {
    T::get_action_from_key(std::declval<typename T::KeyType>())
    } -> std::same_as<typename T::ActionSpace>;
  { T::hash(std::declval<typename T::KeyType>()) } -> std::same_as<std::size_t>;
};

template <environment::EnvironmentType ENVIRON_T>
struct DefaultActionKeymaker
    : StateActionKeymaker<ENVIRON_T,
                          std::pair<typename ENVIRON_T::StateType,
                                    typename ENVIRON_T::ActionSpace>> {

  using baseType =
      StateActionKeymaker<ENVIRON_T,
                          std::pair<typename ENVIRON_T::StateType,
                                    typename ENVIRON_T::ActionSpace>>;

  using EnvironmentType = typename baseType::EnvironmentType;
  using StateType = typename baseType::StateType;
  using ActionSpace = typename baseType::ActionSpace;
  using KeyType = typename baseType::KeyType;

  static KeyType make(const StateType &s, const ActionSpace &action) {
    return std::make_pair(s, action);
  }

  static typename ENVIRON_T::ActionSpace
  get_action_from_key(const std::pair<typename ENVIRON_T::StateType,
                                      typename ENVIRON_T::ActionSpace> &key) {
    return key.second;
  }

  static std::size_t
  hash(const std::pair<typename ENVIRON_T::StateType,
                       typename ENVIRON_T::ActionSpace> &key) {
    return key.first.hash() ^ key.second.hash();
  }

  using Hash = HashBuilder<DefaultActionKeymaker<ENVIRON_T>>;
};

// Helper to print the default key type
template <typename T0, typename T1>
std::ostream &operator<<(std::ostream &os, const std::pair<T0, T1> &p) {
  os << "(" << p.first << ", " << p.second << ")";
  return os;
}
} // namespace policy