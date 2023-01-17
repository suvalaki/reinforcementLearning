#pragma once
#include <iomanip>
#include <iostream>

#include "bandit.hpp"
#include "bandit_environment.hpp"
#include "policy.hpp"

using namespace bandit;

namespace bandit {

template <typename ENVIRON_T>
struct BanditStateActionKeymapper
    : policy::StateActionKeymaker<ENVIRON_T, typename ENVIRON_T::ActionSpace> {

  using baseType =
      policy::StateActionKeymaker<ENVIRON_T, typename ENVIRON_T::ActionSpace>;
  using KeyType = typename baseType::KeyType;
  using StateType = typename baseType::StateType;
  using ActionSpace = typename baseType::ActionSpace;

  static KeyType make(const StateType &s, const ActionSpace &action) {
    return action;
  }

  static ActionSpace get_action_from_key(const KeyType &key) { return key; }

  static std::size_t hash(const KeyType &key) { return key.hash(); }

  using Hash = policy::HashBuilder<BanditStateActionKeymapper<ENVIRON_T>>;
};

template <typename ENVIRON_T>
using GreedyBanditPolicy =
    policy::GreedyPolicy<ENVIRON_T, BanditStateActionKeymapper<ENVIRON_T>>;

} // namespace bandit