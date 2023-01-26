#pragma once

#include <unordered_map>

#include "environment.hpp"
#include "policy/distribution_policy.hpp"

#include "markov_decision_process/finite_state_value_function.hpp"
#include "markov_decision_process/finite_transition_model.hpp"

namespace markov_decision_process {

template <isFiniteStateValueFunction VALUE_FUNCTION_T,
          policy::isDistributionPolicy POLICY_T, auto INITIAL_VALUE = 0.0F,
          auto DISCOUNT_RATE = 0.0F>
typename VALUE_FUNCTION_T::PrecisionType value_from_state_action(
    VALUE_FUNCTION_T &valueFunction,
    const typename VALUE_FUNCTION_T::EnvironmentType &environment,
    const POLICY_T &policy,
    const typename VALUE_FUNCTION_T::EnvironmentType::StateType &state,
    const typename VALUE_FUNCTION_T::EnvironmentType::ActionSpace &action) {

  using EnvironmentType = typename VALUE_FUNCTION_T::EnvironmentType;
  using PrecisionType = typename EnvironmentType::PrecisionType;
  using RewardType = typename EnvironmentType::RewardType;
  using StateType = typename EnvironmentType::StateType;
  using TransitionType = typename EnvironmentType::TransitionType;

  const auto &transitionModel = environment.transitionModel;

  auto reachableStates = environment.getReachableStates(state, action);
  return std::accumulate(
      reachableStates.begin(), reachableStates.end(), 0.0F,
      [&](const auto &value, const auto &nextState) {
        auto transition = TransitionType{state, action, nextState};

        if (transitionModel.find(transition) == transitionModel.end())
          return value;

        return value +
               transitionModel.at(transition) *
                   (RewardType::reward(transition) +
                    valueFunction.discount_rate *
                        valueFunction.valueEstimates
                            .emplace(nextState, valueFunction.initial_value)
                            .first->second);
      });
}

// This mechanism requires the transition model for the finite state
// markov model
template <isFiniteStateValueFunction VALUE_FUNCTION_T,
          policy::isDistributionPolicy POLICY_T, auto INITIAL_VALUE = 0.0F,
          auto DISCOUNT_RATE = 0.0F>
typename VALUE_FUNCTION_T::PrecisionType policy_evaluation_step(
    VALUE_FUNCTION_T &valueFunction,
    const typename VALUE_FUNCTION_T::EnvironmentType &environment,
    const POLICY_T &policy, const typename VALUE_FUNCTION_T::StateType &state) {

  using EnvironmentType = typename VALUE_FUNCTION_T::EnvironmentType;
  using PrecisionType = typename EnvironmentType::PrecisionType;
  using RewardType = typename EnvironmentType::RewardType;
  using StateType = typename EnvironmentType::StateType;
  using TransitionType = typename EnvironmentType::TransitionType;

  const auto &transitionModel = environment.transitionModel;
  auto currentValueEstimate = valueFunction.valueAt(state);

  // For each state action pair reachable from this state evaluate the
  // expected value of the next state given the policy.
  const auto reachableActions = environment.getReachableActions(state);
  auto nextValueEstimate = std::accumulate(
      reachableActions.begin(), reachableActions.end(), 0.0F,
      [&](const auto &value, const auto &action) {
        const auto reachableStates =
            environment.getReachableStates(state, action);
        return value + policy.getProbability(state, {state, action}) *
                           value_from_state_action(valueFunction, environment,
                                                   policy, state, action);
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

template <isFiniteStateValueFunction VALUE_FUNCTION_T,
          policy::isDistributionPolicy POLICY_T, auto INITIAL_VALUE = 0.0F,
          auto DISCOUNT_RATE = 0.0F>
bool policy_improvement_step(
    VALUE_FUNCTION_T &valueFunction,
    const typename VALUE_FUNCTION_T::EnvironmentType &environment,
    POLICY_T &policy,
    const typename VALUE_FUNCTION_T::EnvironmentType::StateType &state) {

  using EnvironmentType = typename VALUE_FUNCTION_T::EnvironmentType;
  using PrecisionType = typename EnvironmentType::PrecisionType;
  using RewardType = typename EnvironmentType::RewardType;
  using StateType = typename EnvironmentType::StateType;
  using TransitionType = typename EnvironmentType::TransitionType;

  using KeyMaker = typename POLICY_T::KeyMaker;

  const auto oldActions = policy.getProbabilities(state);
  const auto oldActionIdx =
      std::max_element(oldActions.begin(), oldActions.end(),
                       [&](const auto &lhs, const auto &rhs) {
                         return lhs.second < rhs.second;
                       });
  const auto oldAction = KeyMaker::get_action_from_key(oldActionIdx->first);

  // For each state action pair find the new distribution of actions
  // under the value function. Take all actions with the argmax and set them
  // to an equal and max probability. When this is a single action the
  // update is effectively deterministic and the probability for taking
  // the action = 1.0

  const auto reachableActions = environment.getReachableActions(state);
  auto nextActionIdx = std::max_element( // warn: under this current approach we
                                         // always pick a single action
      reachableActions.begin(), reachableActions.end(),
      [&](const auto &lhs, const auto &rhs) {
        return value_from_state_action(valueFunction, environment, policy,
                                       state, lhs) <
               value_from_state_action(valueFunction, environment, policy,
                                       state, rhs);
      });
  const auto nextAction = *nextActionIdx;
  const auto policyStable = oldAction == nextAction;

  // update the policy - by setting the policy to be deterministic on the
  // new argmax
  for (const auto &action : reachableActions) {
    if (action == nextAction)
      policy.setProbability(state, {state, action}, 1.0F);
    policy.setProbability(state, {state, action}, 0.0F);
  }

  return policyStable;
}

template <isFiniteStateValueFunction VALUE_FUNCTION_T,
          policy::isDistributionPolicy POLICY_T, auto INITIAL_VALUE = 0.0F,
          auto DISCOUNT_RATE = 0.0F>
void policy_improvement(
    VALUE_FUNCTION_T &valueFunction,
    const typename VALUE_FUNCTION_T::EnvironmentType &environment,
    POLICY_T &policy, const typename VALUE_FUNCTION_T::PrecisionType &epsilon) {

  bool policyStable = false;

  // While any of the policies are not stable keep improving them
  do {
    policy_evaluation(valueFunction, environment, policy, epsilon);
    const auto allStates = environment.getAllPossibleStates();
    for (const auto &state : allStates) {
      policyStable |=
          policy_improvement_step(valueFunction, environment, policy, state);
    }
  } while (not policyStable);
}

} // namespace markov_decision_process
