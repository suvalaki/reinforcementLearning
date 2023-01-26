#pragma once

#include <unordered_map>

#include "environment.hpp"
#include "policy/distribution_policy.hpp"

#include "markov_decision_process/finite_state_value_function.hpp"
#include "markov_decision_process/finite_transition_model.hpp"

namespace markov_decision_process {

// This mechanism requires the transition model for the finite state
// markov model
template <isFiniteStateValueFunction VALUE_FUNCTION_T,
          policy::isDistributionPolicy POLICY_T, auto INITIAL_VALUE = 0.0F,
          auto DISCOUNT_RATE = 0.0F>
typename VALUE_FUNCTION_T::PrecisionType policy_evaluation_step(
    VALUE_FUNCTION_T &valueFunction,
    const typename VALUE_FUNCTION_T::EnvironmentType &environment,
    POLICY_T &policy, const typename VALUE_FUNCTION_T::StateType &state) {

  using EnvironmentType = typename VALUE_FUNCTION_T::EnvironmentType;
  using PrecisionType = typename EnvironmentType::PrecisionType;
  using RewardType = typename EnvironmentType::RewardType;
  using StateType = typename EnvironmentType::StateType;
  using TransitionType = typename EnvironmentType::TransitionType;

  const auto &transitionModel = environment.transitionModel;
  auto currentValueEstimate = valueFunction.valueAt(state);
  auto nextValueEstimate = 0.0F;

  // For each state action pair reachable from this state evaluate the
  // expected value of the next state given the policy.
  const auto reachableActions = environment.getReachableActions(state);
  nextValueEstimate += std::accumulate(
      reachableActions.begin(), reachableActions.end(), 0.0F,
      [&](const auto &value, const auto &action) {
        const auto reachableStates =
            environment.getReachableStates(state, action);
        return value +
               policy.getProbability(state, {state, action}) *
                   std::accumulate(
                       reachableStates.begin(), reachableStates.end(), 0.0F,
                       [&](const auto v, const auto &nextState) {
                         auto transition =
                             TransitionType{state, action, nextState};

                         if (transitionModel.find(transition) ==
                             transitionModel.end())
                           return v;

                         return v +
                                transitionModel.at(transition) *
                                    (RewardType::reward(transition) +
                                     valueFunction.discount_rate *
                                         valueFunction.valueEstimates
                                             .emplace(
                                                 nextState,
                                                 valueFunction.initial_value)
                                             .first->second);
                       });
      });

  return nextValueEstimate;
}

template <isFiniteStateValueFunction VALUE_FUNCTION_T,
          policy::isDistributionPolicy POLICY_T, auto INITIAL_VALUE = 0.0F,
          auto DISCOUNT_RATE = 0.0F>
void policy_evaluation(
    VALUE_FUNCTION_T &valueFunction,
    const typename POLICY_T::EnvironmentType &environment, POLICY_T &policy,
    const typename VALUE_FUNCTION_T::PrecisionType &epsilon) {

  typename VALUE_FUNCTION_T::PrecisionType delta = 0.0F;
  // sweep over all states and update the value function. When finally no
  // states change significantly we have converged and can exit
  do {
    delta = 0.0F;
    for (const auto &state : environment.getAllPossibleStates()) {
      auto oldValue = valueFunction.valueAt(state);
      auto newValue =
          policy_evaluation_step(valueFunction, environment, policy, state);
      delta = std::max(delta, std::abs(oldValue - newValue));
      valueFunction.valueEstimates.at(state) = newValue;
    }
  } while (delta > epsilon);
}

} // namespace markov_decision_process
