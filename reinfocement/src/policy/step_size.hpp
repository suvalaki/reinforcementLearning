#pragma once
#include <random>
#include <type_traits>
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

template <auto X> inline constexpr bool admissible_is_between() {
  return 0.0F < X and 1.0F > X;
};

template <auto STEP_S>
concept allowed_step_size = admissible_is_between<STEP_S>() &&
    std::is_floating_point<decltype(STEP_S)>::value;

template <isStateActionValue VALUE_T, auto STEP_S>
requires allowed_step_size<STEP_S>
struct constant_step_size_taker {
  using StateValueType = VALUE_T;
  using PrecisionType = typename StateValueType::PrecisionType;
  static typename StateValueType::PrecisionType
  getStepSize(const StateValueType &value) {
    return STEP_S;
  }
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