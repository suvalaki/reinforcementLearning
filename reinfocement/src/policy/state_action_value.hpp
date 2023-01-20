#pragma once
#include "environment.hpp"
#include <cmath>
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
  PrecisionType value = 0;
  // number of steps taken for this state-action estimate
  std::size_t step = 0;

  StateActionValue(const PrecisionType &value = 0, const std::size_t &step = 0)
      : value(value), step(step) {}

  bool operator<(const StateActionValue &other) const {
    return value < other.value;
  }

  virtual void noFocusUpdate() {}
};

template <typename T>
concept isStateActionValue =
    std::is_base_of_v<StateActionValue<typename T::EnvironmentType>, T> ||
    std::is_same_v<StateActionValue<typename T::EnvironmentType>, T>;

template <environment::EnvironmentType ENVIRON_T, auto DEGREE_OF_EXPLORATION>
struct UpperConfidenceBoundStateActionValue
    : public StateActionValue<ENVIRON_T> {

  static_assert(DEGREE_OF_EXPLORATION > 0.0F,
                "Degree of exploration must be greater than 0.0F");

  using EnvironmentType = ENVIRON_T;
  using PrecisionType = typename ENVIRON_T::PrecisionType;
  using StateActionValueType = StateActionValue<ENVIRON_T>;
  using StateActionValueType::step;
  using StateActionValueType::value;

  static float constexpr degree_of_exploration = DEGREE_OF_EXPLORATION;

  // total count over all steps over all actions.
  std::size_t total_step = 0;

  UpperConfidenceBoundStateActionValue(const PrecisionType &value = 0,
                                       const std::size_t &step = 0,
                                       const std::size_t &total_step = 0)
      : StateActionValueType{value, step}, total_step(total_step) {}

  PrecisionType getUpperConfidenceBound() const {
    return step == 0 ? value
                     : value + degree_of_exploration *
                                   std::sqrt(std::log(this->total_step) / step);
  }

  bool operator<(const UpperConfidenceBoundStateActionValue &other) const {
    return getUpperConfidenceBound() < other.getUpperConfidenceBound();
  }

  virtual void noFocusUpdate() { total_step++; }
};

} // namespace policy