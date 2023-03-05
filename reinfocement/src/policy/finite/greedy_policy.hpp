#pragma once

#include "environment.hpp"
#include "policy/finite/policy.hpp"
#include "policy/finite/value_policy.hpp"
#include "policy/greedy_policy.hpp"
#include "policy/objectives/finite_value.hpp"
#include "policy/objectives/finite_value_function.hpp"
#include "policy/objectives/step_size.hpp"

#define GPT FiniteGreedyPolicy<VALUE_FUNCTION_T>

namespace policy {

// Just need to override the Argmax function to use the Q-Table since this is a finite value function
// Or at least it is after we add the appropriate mixin.
// How to handle updates via step size takers? - Unknown at this time - Probably need to think of a way
// of applying incremental updates to ANY key type Q-table first.

template <objectives::isFiniteValueFunction VALUE_FUNCTION_T>
struct FiniteGreedyPolicy : GreedyPolicy<VALUE_FUNCTION_T>, FinitePolicyValueFunctionMixin<VALUE_FUNCTION_T> {

  SETUP_TYPES_FROM_NESTED_ENVIRON(SINGLE_ARG(VALUE_FUNCTION_T::EnvironmentType));
  using ValueFunctionType = VALUE_FUNCTION_T;
  using ValueFunctionBaseType = typename ValueFunctionType::ValueFunctionBaseType;
  using StepSizeTaker = typename VALUE_FUNCTION_T::StepSizeTaker;
  using KeyMaker = typename VALUE_FUNCTION_T::KeyMaker;
  using KeyType = typename KeyMaker::KeyType;
  using ValueType = typename VALUE_FUNCTION_T::ValueType;
  using ValueTableType = typename objectives::FiniteValueFunction<ValueFunctionType>::ValueTableType;

  void update(const EnvironmentType &e, const TransitionType &s) override;
  // ActionSpace operator()(const EnvironmentType &e, const StateType &s) const override { return ActionSpace{}; };
  using ValueFunctionType::initialize;

  FiniteGreedyPolicy(auto &&...args)
      : FinitePolicyValueFunctionMixin<VALUE_FUNCTION_T>(args...), GreedyPolicy<VALUE_FUNCTION_T>(args...) {}
  FiniteGreedyPolicy(const FiniteGreedyPolicy &p)
      : FinitePolicyValueFunctionMixin<VALUE_FUNCTION_T>(p), GreedyPolicy<VALUE_FUNCTION_T>(p) {}
  FiniteGreedyPolicy &operator=(FiniteGreedyPolicy &&g) {
    ValueFunctionType(std::move(g));
    return *this;
  }

  ActionSpace operator()(const EnvironmentType &e, const StateType &s) const override {
    return FinitePolicyValueFunctionMixin<VALUE_FUNCTION_T>::getArgmaxAction(e, s);
  };
};

template <objectives::isFiniteValueFunction VALUE_FUNCTION_T>
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
using FiniteGreedyPolicyC = FiniteGreedyPolicy<
    objectives::FiniteValueFunction<objectives::ValueFunction<KEYMAKER_C<E>, VALUE_C<E>, INITIAL_VALUE, DISCOUNT_RATE>,
                                    INCREMENTAL_STEPSIZE_C<VALUE_C<E>>>>;

} // namespace policy

#undef GPT