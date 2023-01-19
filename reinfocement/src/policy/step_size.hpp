#pragma once
#include <random>
#include <xtensor/xfixed.hpp>
#include <xtensor/xrandom.hpp>

#include "action.hpp"
#include "environment.hpp"
#include "spec.hpp"
#include "state_action_value.hpp"

namespace policy {

template <typename T>
concept step_size_taker = requires(T t) {
  typename T::StateValueType;
  {
    T::getStepSize(std::declval<typename T::StateValueType>())
    } -> std::same_as<typename T::PrecisionType>;
};

template <isStateActionValue VALUE_T> struct weighted_average_step_size_taker {
  using StateValueType = VALUE_T;
  using PrecisionType = typename StateValueType::PrecisionType;
  static typename StateValueType::PrecisionType
  getStepSize(const StateValueType &value) {
    return 1.0 / (value.step + 1);
  }
};
} // namespace policy