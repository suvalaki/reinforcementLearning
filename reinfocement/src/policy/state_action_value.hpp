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

template <typename T>
concept isStateActionValueFactory = requires(T t) {
  { t.create() } -> std::same_as<typename T::ValueType>;
  { t.update() } -> std::same_as<void>;
};

template <environment::EnvironmentType ENVIRON_T> struct StateActionValue {
  SETUP_TYPES_FROM_ENVIRON(SINGLE_ARG(ENVIRON_T))
  // Current state-action-value estimatae
  PrecisionType value = 0;
  // number of steps taken for this state-action estimate
  std::size_t step = 0;

  StateActionValue(const PrecisionType &value = 0, const std::size_t &step = 0) : value(value), step(step) {}

  bool operator<(const StateActionValue &other) const { return value < other.value; }

  bool operator<(const PrecisionType &other) const { return value < other; }
  bool operator>(const PrecisionType &other) const { return value > other; }
  bool operator==(const PrecisionType &other) const { return value == other; }

  bool operator==(const StateActionValue &other) const { return value == other.value && step == other.step; }

  virtual void noFocusUpdate() {}

  struct Factory {
    using ValueType = StateActionValue;
    StateActionValue create(const PrecisionType &value = 0, const std::size_t &step = 0) {
      return StateActionValue{value, step};
    }

    void update() {}
  };
};

template <typename T>
concept isStateActionValue = std::is_base_of_v<StateActionValue<typename T::EnvironmentType>, T> &&
    isStateActionValueFactory<typename T::Factory>;

template <environment::EnvironmentType ENVIRON_T, auto DEGREE_OF_EXPLORATION>
struct UpperConfidenceBoundStateActionValue : public StateActionValue<ENVIRON_T> {

  static_assert(DEGREE_OF_EXPLORATION > 0.0F, "Degree of exploration must be greater than 0.0F");

  SETUP_TYPES_FROM_ENVIRON(SINGLE_ARG(ENVIRON_T))
  using StateActionValueType = StateActionValue<ENVIRON_T>;
  using StateActionValueType::step;
  using StateActionValueType::value;

  static float constexpr degree_of_exploration = DEGREE_OF_EXPLORATION;

  // total count over all steps over all actions.
  std::size_t &total_step;

  UpperConfidenceBoundStateActionValue(const PrecisionType &value, const std::size_t &step, std::size_t &total_step)
      : StateActionValueType{value, step}, total_step(total_step) {}

  PrecisionType getUpperConfidenceBound() const {
    return step == 0 ? value : value + degree_of_exploration * std::sqrt(std::log(this->total_step) / step);
  }

  bool operator<(const UpperConfidenceBoundStateActionValue &other) const {
    return getUpperConfidenceBound() < other.getUpperConfidenceBound();
  }

  bool operator==(const UpperConfidenceBoundStateActionValue &other) const {
    return value == other.value && step == other.step && total_step == other.total_step;
  }

  virtual void noFocusUpdate() { total_step++; }

  struct Factory {

    using ValueType = UpperConfidenceBoundStateActionValue;
    // total count over all steps over all actions.
    std::size_t total_step = 0;

    UpperConfidenceBoundStateActionValue create(const PrecisionType &value = 0, const std::size_t &step = 0) {
      return UpperConfidenceBoundStateActionValue{value, step, this->total_step};
    }

    void update() { total_step++; }
  };
};

} // namespace policy