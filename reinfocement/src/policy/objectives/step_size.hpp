#pragma once
#include <random>
#include <type_traits>
#include <xtensor/xfixed.hpp>
#include <xtensor/xrandom.hpp>

#include "action.hpp"
#include "dummy_environment.hpp"
#include "environment.hpp"
#include "policy/objectives/finite_value.hpp"
#include "spec.hpp"

namespace policy::objectives {

// Step size defines how the monte carlo updates proceed.
// https://goodboychan.github.io/reinforcement_learning/2020/06/16/03-Monte-Carlo-Policy-Evaluation.html

/**
 * @brief A concept that checks if a type is a step size taker. A step size
 * taker is a TMP struct that runs getStepSize function on a statevalue type and
 * returns a variable sized step size. The step size alpha ensures convergence
 * of the action-value when sum a(t, a) -> \infty and sum a(t, a)^2 < infty.
 *
 * @tparam T
 */
template <typename T>
concept step_size_taker = requires(T t) {
  typename T::StateValueType;
  { T::getStepSize(std::declval<typename T::StateValueType>()) } -> std::same_as<typename T::PrecisionType>;
};

template <auto X>
inline constexpr bool admissible_is_between() {
  return 0.0F < X and 1.0F > X;
};

/**
 * @brief A concept that checks if the step size is a floating point type and
 * between 0 and 1. Steps must be between 0 and 1 to that the value always
 * changes and is an admissible weighted average.
 *
 * @tparam STEP_S
 */
template <auto STEP_S>
concept allowed_step_size = admissible_is_between<STEP_S>() && std::is_floating_point<decltype(STEP_S)>::value;

template <typename T>
concept isStepSizeTaker = requires(T t) {
  typename T::StateValueType;
  typename T::PrecisionType;
  { T::getStepSize(std::declval<typename T::StateValueType>()) } -> std::same_as<typename T::PrecisionType>;
};

/**
 * @brief A step size taker that always moves with a constant step size. It is
 * the weighted average of the current value and the difference between the new
 * value and the current value. The step size denotes the weight towards the new
 * estimate for the value.
 *
 * @tparam VALUE_T
 * @tparam STEP_S
 */
template <isFiniteValue VALUE_T, auto STEP_S = 0.5>
requires allowed_step_size<STEP_S>
struct constant_step_size_taker {
  using StateValueType = VALUE_T;
  using PrecisionType = typename StateValueType::PrecisionType;
  static typename StateValueType::PrecisionType getStepSize(const StateValueType &value) { return STEP_S; }
};

template <template <typename> typename T>
concept isConstantStepSizeTemplate = std::is_same_v<
    T<FiniteValue<environment::dummy::DummyFiniteEnvironment>>,
    constant_step_size_taker<FiniteValue<environment::dummy::DummyFiniteEnvironment>>>;

template <isFiniteValue VALUE_T>
struct weighted_average_step_size_taker {
  // sample average step size
  using StateValueType = VALUE_T;
  using PrecisionType = typename StateValueType::PrecisionType;
  static typename StateValueType::PrecisionType getStepSize(const StateValueType &value) {
    return 1.0 / (value.step + 1);
  }
};

template <template <typename> typename T>
concept isWeightedAverageStepSizeTemplate = std::is_same_v<
    T<FiniteValue<environment::dummy::DummyFiniteEnvironment>>,
    weighted_average_step_size_taker<FiniteValue<environment::dummy::DummyFiniteEnvironment>>>;

template <template <typename> typename T>
concept isStepSizeTemplate = isConstantStepSizeTemplate<T> || isWeightedAverageStepSizeTemplate<T>;

} // namespace policy::objectives
