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

/**
 * @brief A Finite version of a greedy policy. The value function is expected to be tabular and able to be
 * parsed. As such the selection criteria for policy action is just to iterate over the table and return the
 * key associated with the highest value.
 *
 * @tparam VALUE_FUNCTION_T Finite value function.
 */
template <objectives::isFiniteValueFunction VALUE_FUNCTION_T>
struct FiniteGreedyPolicy : GreedyPolicy<VALUE_FUNCTION_T>, FinitePolicyValueFunctionMixin<VALUE_FUNCTION_T> {

  SETUP_TYPES_W_VALUE_FUNCTION(VALUE_FUNCTION_T);
  using ValueFunctionBaseType = typename ValueFunctionType::ValueFunctionBaseType;
  using StepSizeTaker = typename VALUE_FUNCTION_T::StepSizeTaker;
  using ValueTableType = typename objectives::FiniteValueFunction<ValueFunctionType>::ValueTableType;

  FiniteGreedyPolicy(auto &&...args);
  FiniteGreedyPolicy(const FiniteGreedyPolicy &p);

  using ValueFunctionType::initialize;
  void update(const EnvironmentType &e, const TransitionType &s) override;
  ActionSpace operator()(const EnvironmentType &e, const StateType &s) const override;
};

template <objectives::isFiniteValueFunction VALUE_FUNCTION_T>
GPT::FiniteGreedyPolicy(auto &&...args)
    : FinitePolicyValueFunctionMixin<VALUE_FUNCTION_T>(args...), GreedyPolicy<VALUE_FUNCTION_T>(args...) {}

template <objectives::isFiniteValueFunction VALUE_FUNCTION_T>
GPT::FiniteGreedyPolicy(const FiniteGreedyPolicy &p)
    : FinitePolicyValueFunctionMixin<VALUE_FUNCTION_T>(p), GreedyPolicy<VALUE_FUNCTION_T>(p) {}

template <objectives::isFiniteValueFunction VALUE_FUNCTION_T>
void GPT::update(const EnvironmentType &e, const TransitionType &s) {
  this->incrementalUpdate(e, s);
}

/** @brief Take the argmax action from the value function table */
template <objectives::isFiniteValueFunction VALUE_FUNCTION_T>
auto GPT::operator()(const EnvironmentType &e, const StateType &s) const -> ActionSpace {
  return FinitePolicyValueFunctionMixin<VALUE_FUNCTION_T>::getArgmaxAction(e, s);
};

/** @brief Helper template to enable generation of a default Finite Greedy Policy. */
template <
    environment::FiniteEnvironmentType E,
    template <typename>
    typename KEYMAKER_C,
    template <typename> typename VALUE_C = objectives::FiniteValue,
    template <typename> typename INCREMENTAL_STEPSIZE_C = objectives::weighted_average_step_size_taker,
    auto INITIAL_VALUE = 0.0F,
    auto DISCOUNT_RATE = 0.0F>
using FiniteGreedyPolicyC = FiniteGreedyPolicy<objectives::FiniteValueFunction<
    objectives::ValueFunction<KEYMAKER_C<E>, VALUE_C<E>, INITIAL_VALUE, DISCOUNT_RATE>,
    INCREMENTAL_STEPSIZE_C<VALUE_C<E>>>>;

} // namespace policy

#undef GPT