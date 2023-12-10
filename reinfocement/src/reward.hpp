#pragma once

#include <array>
#include <cmath>
#include <type_traits>

#include "action.hpp"
#include "spec.hpp"
#include "state.hpp"
#include "step.hpp"
#include "transition.hpp"

namespace reward {

using action::ActionType;
using step::Step;
using transition::Transition;
using transition::TransitionSequence;

template <typename... T>
struct Reward {};
template <ActionType ACTION0>
struct Reward<ACTION0> {
  using StateType = typename ACTION0::StateType;
  using PrecisionType = typename StateType::PrecisionType;
  using ActionSpace = ACTION0;
  using ActionSpecType = typename ActionSpace::SpecType;
  using StepType = Step<ActionSpace>;
  using TransitionType = Transition<ActionSpace>;
  static PrecisionType reward(const TransitionType &t) { return 0.0F; }
};

template <typename REWARD_T>
concept RewardBaseType = std::is_base_of_v<Reward<typename REWARD_T::ActionSpace>, REWARD_T>;

template <typename REWARD_T>
concept RewardProtocol = requires(REWARD_T t, const typename REWARD_T::TransitionType &tn) {
  // Reqyure that the Object implementing the reward protocol to implement
  // static PrecisionType reward(const TransitionType& t) {return 0.0F;}
  { REWARD_T::reward(tn) };
  // Require Precision Type is inside REWARD_D
  typename REWARD_T::PrecisionType;
  // Require ActionSpace is inside REWARD_D - Rewards always act over an action
  // space
  typename REWARD_T::ActionSpace;
};

template <typename T>
concept RewardType = RewardBaseType<T> && RewardProtocol<T>;

// Return differs from reward. Reward is immediate whilst return discounts
// the future (or estimated future) under a given action
template <RewardProtocol REWARD_T>
struct Return {

  using PrecisionType = typename REWARD_T::PrecisionType;
  using RewardType = REWARD_T;
  using ActionSpace = typename RewardType::ActionSpace;
  using ActionSpecType = typename ActionSpace::SpecType;

  // Discounted Return of future SEQUENCE_LENGTH time steps
  template <std::size_t SEQUENCE_LENGTH>
  static PrecisionType value(
      const TransitionSequence<SEQUENCE_LENGTH, typename RewardType::ActionSpace> &t,
      const PrecisionType &discountRate = 1.0F) {
    auto arr = future_value(t, discountRate);
    return std::accumulate(arr.begin(), arr.end(), static_cast<PrecisionType>(0));
  }

  // Return raw rewards for future time steps without discounting or any other
  // calcs
  template <std::size_t SEQUENCE_LENGTH>
  static std::array<PrecisionType, SEQUENCE_LENGTH>
  future_value(const TransitionSequence<SEQUENCE_LENGTH, ActionSpace> &t, const PrecisionType &discountRate = 1.0F) {
    std::array<PrecisionType, SEQUENCE_LENGTH> result;
    for (int i = 0; i < SEQUENCE_LENGTH; i++) {
      result[i] = std::pow(discountRate, static_cast<PrecisionType>(i)) * RewardType::reward(t[i]);
    }
    return result;
  }
};

} // namespace reward
