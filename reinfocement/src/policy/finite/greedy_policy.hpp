#pragma once

#include "environment.hpp"
#include "policy/finite/policy.hpp"
#include "policy/finite/value_policy.hpp"
#include "policy/greedy_policy.hpp"
#include "policy/objectives/finite_value.hpp"
#include "policy/objectives/finite_value_function.hpp"
#include "policy/objectives/step_size.hpp"

#define GPT FiniteGreedyPolicy<KEYMAPPER_T, VALUE_T, INITIAL_VALUE, DISCOUNT_RATE, INCREMENTAL_STEPSIZE_T>

namespace policy {

// Just need to override the Argmax function to use the Q-Table since this is a finite value function
// Or at least it is after we add the appropriate mixin.
// How to handle updates via step size takers? - Unknown at this time - Probably need to think of a way
// of applying incremental updates to ANY key type Q-table first.

template <objectives::isValueFunctionKeymaker KEYMAPPER_T,
          objectives::isFiniteValue VALUE_T = objectives::FiniteValue<typename KEYMAPPER_T::EnvironmentType>,
          auto INITIAL_VALUE = 0.0F,
          auto DISCOUNT_RATE = 0.0F,
          objectives::isStepSizeTaker INCREMENTAL_STEPSIZE_T = objectives::weighted_average_step_size_taker<VALUE_T>>
requires environment::FiniteEnvironmentType<typename KEYMAPPER_T::EnvironmentType>
struct FiniteGreedyPolicy
    : FinitePolicyValueFunctionMixin<GreedyPolicy<KEYMAPPER_T, VALUE_T, INITIAL_VALUE, DISCOUNT_RATE>,
                                     INCREMENTAL_STEPSIZE_T> {

  SETUP_TYPES_FROM_NESTED_ENVIRON(SINGLE_ARG(KEYMAPPER_T::EnvironmentType));
  using ValueFunctionBaseType = GreedyPolicy<KEYMAPPER_T, VALUE_T, INITIAL_VALUE, DISCOUNT_RATE>;
  using StepSizeTaker = INCREMENTAL_STEPSIZE_T;
  using ValueFunctionType =
      FinitePolicyValueFunctionMixin<GreedyPolicy<KEYMAPPER_T, VALUE_T, INITIAL_VALUE, DISCOUNT_RATE>,
                                     INCREMENTAL_STEPSIZE_T>;
  using KeyMaker = KEYMAPPER_T;
  using KeyType = typename KeyMaker::KeyType;
  using ValueType = VALUE_T;
  using ValueTableType = typename objectives::FiniteValueFunction<ValueFunctionType>::ValueTableType;

  void update(const EnvironmentType &e, const TransitionType &s) override;
  // ActionSpace operator()(const EnvironmentType &e, const StateType &s) const override { return ActionSpace{}; };
};

template <objectives::isValueFunctionKeymaker KEYMAPPER_T,
          objectives::isFiniteValue VALUE_T,
          auto INITIAL_VALUE,
          auto DISCOUNT_RATE,
          objectives::isStepSizeTaker INCREMENTAL_STEPSIZE_T>
void GPT::update(const EnvironmentType &e, const TransitionType &s) {
  this->incrementalUpdate(e, s);
}

// Some useful typedefs to bootstrap different keymakers for the greedy policy.
template <environment::FiniteEnvironmentType E,
          template <typename>
          typename KEYMAKER_C,
          template <typename> typename VALUE_C = objectives::FiniteValue,
          template <typename> typename INCREMENTAL_STEPSIZE_C = objectives::weighted_average_step_size_taker,
          auto INITIAL_VALUE = 0.0F,
          auto DISCOUNT_RATE = 0.0F>
using FiniteGreedyPolicyC =
    FiniteGreedyPolicy<KEYMAKER_C<E>, VALUE_C<E>, INITIAL_VALUE, DISCOUNT_RATE, INCREMENTAL_STEPSIZE_C<VALUE_C<E>>>;

} // namespace policy

#undef GPT