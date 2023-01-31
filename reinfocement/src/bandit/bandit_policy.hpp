#pragma once
#include <iomanip>
#include <iostream>

#include "bandit.hpp"
#include "bandit_environment.hpp"
#include "policy/distribution_policy.hpp"
#include "policy/epsilon_greedy_policy.hpp"
#include "policy/greedy_policy.hpp"
#include "policy/policy.hpp"
#include "policy/random_policy.hpp"

using namespace bandit;

namespace bandit {

template <typename ENVIRON_T>
struct BanditStateActionKeymapper
    : policy::StateActionKeymaker<ENVIRON_T, typename ENVIRON_T::ActionSpace> {

  SETUP_KEYMAKER_TYPES(SINGLE_ARG(
      policy::StateActionKeymaker<ENVIRON_T, typename ENVIRON_T::ActionSpace>));

  static KeyType make(const StateType &s, const ActionSpace &action) {
    return action;
  }

  static StateType get_state_from_key(const EnvironmentType &e,
                                      const KeyType &key) {
    return e.getNullState();
  }

  static ActionSpace get_action_from_key(const KeyType &key) { return key; }

  static std::size_t hash(const KeyType &key) { return key.hash(); }

  using Hash = policy::HashBuilder<BanditStateActionKeymapper<ENVIRON_T>>;
};

template <typename ENVIRON_T>
using BanditStateActionValue = ::policy::StateActionValue<ENVIRON_T>;

template <typename ENVIRON_T>
using BanditWeightedAverageStepSizeTaker =
    ::policy::weighted_average_step_size_taker<
        BanditStateActionValue<ENVIRON_T>>;

template <typename ENVIRON_T>
using GreedyBanditPolicy = ::policy::GreedyPolicy< //
    ENVIRON_T,                                     //
    BanditStateActionKeymapper<ENVIRON_T>,         //
    BanditStateActionValue<ENVIRON_T>,             //
    BanditWeightedAverageStepSizeTaker<ENVIRON_T>  //
    >;

template <typename ENVIRON_T, auto DEGREE_OF_EXPLORATION>
using BanditUpperCBoundStateActionValue =
    ::policy::UpperConfidenceBoundStateActionValue<ENVIRON_T,
                                                   DEGREE_OF_EXPLORATION>;

template <typename ENVIRON_T, auto DEGREE_OF_EXPLORATION>
using UpperConfidenceBoundGreedyBanditPolicy = ::policy::GreedyPolicy<   //
    ENVIRON_T,                                                           //
    BanditStateActionKeymapper<ENVIRON_T>,                               //
    BanditUpperCBoundStateActionValue<ENVIRON_T, DEGREE_OF_EXPLORATION>, //
    BanditWeightedAverageStepSizeTaker<ENVIRON_T>                        //
    >;

template <typename ENVIRON_T>
using BanditDistributionStateActionValue =
    ::policy::DistributionStateActionValue<ENVIRON_T>;

template <typename ENVIRON_T>
using BanditDistributionPolicy = ::policy::DistributionPolicy< //
    ENVIRON_T,                                                 //
    BanditStateActionKeymapper<ENVIRON_T>,                     //
    BanditDistributionStateActionValue<ENVIRON_T>,             //
    BanditWeightedAverageStepSizeTaker<ENVIRON_T>              //
    >;

} // namespace bandit