#pragma once
#include "environment.hpp"
#include <random>
#include <xtensor/xfixed.hpp>
#include <xtensor/xrandom.hpp>

#include "action.hpp"
#include "environment.hpp"
#include "spec.hpp"

namespace policy {

template <environment::EnvironmentType ENVIRON_T> struct StateActionValue {
  using EnvironmentType = ENVIRON_T;
  using PrecisionType = typename ENVIRON_T::PrecisionType;
  // Current state-action-value estimatae
  PrecisionType value;
  // number of steps taken for this state-action estimate
  std::size_t step;
};

template <typename T>
concept isStateActionValue =
    std::is_base_of_v<StateActionValue<typename T::EnvironmentType>, T> ||
    std::is_same_v<StateActionValue<typename T::EnvironmentType>, T>;

} // namespace policy