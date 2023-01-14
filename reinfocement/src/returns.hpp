#pragma once

#include <array>
#include <type_traits>

#include "action.hpp"
#include "reward.hpp"
#include "spec.hpp"
#include "state.hpp"
#include "step.hpp"
#include "transition.hpp"

namespace returns {

using action::ActionType;
using reward::RewardProtocol;
using step::Step;
using transition::Transition;
using transition::TransitionSequence;

// Return differs from reward. Reward is immediate whilst return discounts
// the future (or estimated future) under a given action
template <RewardProtocol REWARD_T> struct Return {

  using PrecisionType = REWARD_T::PrecisionType;
  using RewardType = REWARD_T;
  using ActionSpace = typename RewardType::ActionSpace;
  using ActionSpecType = typename ActionSpace::SpecType;

  // Discounted Return of future SEQUENCE_LENGTH time steps
  template <std::size_t SEQUENCE_LENGTH>
  static PrecisionType
  value(const TransitionSequence<SEQUENCE_LENGTH,
                                 typename RewardType::ActionSpace> &t) {
    auto arr = future_value(t);
    return std::accumulate(arr.begin(), arr.end(),
                           static_cast<PrecisionType>(0));
  }

  // Return raw rewards for future time steps without discounting or any other
  // calcs
  template <std::size_t SEQUENCE_LENGTH>
  static std::array<PrecisionType, SEQUENCE_LENGTH>
  future_value(const TransitionSequence<SEQUENCE_LENGTH, ActionSpace> &t) {
    std::array<PrecisionType, SEQUENCE_LENGTH> result;
    for (int i = 0; i < SEQUENCE_LENGTH; i++) {
      result[i] = RewardType::reward(t[i]);
    }
    return result;
  }
};

template <typename RETURN_T>
concept ReturnType =
    std::is_base_of_v<Return<typename RETURN_T::RewardType>, RETURN_T>;

template <RewardProtocol REWARD_T, float DISCOUNT = 0.8F>
struct DiscountReturn : Return<REWARD_T> {

  using BaseType = Return<REWARD_T>;
  constexpr static REWARD_T::PrecisionType discountRate = DISCOUNT;

  template <std::size_t SEQUENCE_LENGTH>
  static BaseType::PrecisionType value(
      const TransitionSequence<SEQUENCE_LENGTH,
                               typename BaseType::RewardType::ActionSpace> &t) {
    auto arr = BaseType::future_value(t);
    typename BaseType::PrecisionType result = 0.0F;
    for (int i = 0; i < SEQUENCE_LENGTH; i++) {
      result += arr[i] * static_cast<typename BaseType::PrecisionType>(
                             std::pow(discountRate, i));
    }
    return result;
  }
};

} // namespace returns
