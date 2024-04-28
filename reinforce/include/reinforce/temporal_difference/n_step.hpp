#pragma once

#include "reinforce/environment.hpp"

namespace temporal_difference {

template <environment::EnvironmentType E>
auto nStepReturn(const E &environment, const typename E::PrecisionType &discountRate, const auto &...transitions) ->
    typename E::PrecisionType {
  using RewardType = typename E::RewardType;
  using ReturnType = typename E::ReturnType;
  using TransitionSequenceType =
      environment::TransitionSequence<sizeof...(transitions), typename RewardType::ActionSpace>;
  return ReturnType::value(TransitionSequenceType{transitions...}, discountRate);
}

} // namespace temporal_difference