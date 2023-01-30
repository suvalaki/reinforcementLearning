#pragma once

#include <algorithm>

#include "markov_decision_process/policy_iteration.hpp"
#include "policy/value.hpp"

// Value iteration algorithm for solving MDPs.
// A drawback of policy iteration is that it requires a policy evaluation step
// which is itself iterative. While policy evaluation converges in the limit to
// the value we can stop evaluation early and use the current value estimate to
// improve the policy. This is the idea behind value iteration. We can truncate
// the policy evaluation step in a number of ways without losing the convergence
// garantuee.
//
// (i) The simplest way is to perform a fixed number of iterations. When
// the fixed number is a single iteration we get "value itreation". This can be
// thought of as a single update of the Bellman optimality equation.
//
// The update step is:
// \begin{equation} v_{k+1}(s)
//  = max_{a} E[R_{t+1} + \gamma v_k(S_{t+1})|S_t=s,A_t=a]
//  = \max_{a} \sum_{s'} p(s'|s,a) (r(s,a,s') + \gamma v_k(s')) \end{equation}
// The difference between this update is that we DONT consider other actions in
// our assessment and always update to the maximum value. This simplifies the
// amount of searching we need to do.
//
// The policy update step for value iteration is to simply take the action which
// results in this max
// \begin{equation} \pi_{k+1}(s) = argmax_{a} \sum_{s'} p(s'|s,a)[r + \gamma
// V(s')] \end{equation}
namespace markov_decision_process::value_iteration {

/**
 * @brief for a single state perform a step of the Belman optimality equation
 * keeping only the max value - rather than the complete expectation over all
 * possible values.
 */
template <policy::isFiniteStateValueFunction VALUE_FUNCTION_T>
typename VALUE_FUNCTION_T::PrecisionType value_iteration_policy_estimation_step(
    VALUE_FUNCTION_T &valueFunction,
    const typename VALUE_FUNCTION_T::EnvironmentType &environment,
    const typename VALUE_FUNCTION_T::StateType &state) {

  using EnvironmentType = typename VALUE_FUNCTION_T::EnvironmentType;
  using PrecisionType = typename EnvironmentType::PrecisionType;
  using RewardType = typename EnvironmentType::RewardType;
  using StateType = typename EnvironmentType::StateType;
  using TransitionType = typename EnvironmentType::TransitionType;

  const auto &transitionModel = environment.transitionModel;
  auto currentValueEstimate = valueFunction.valueAt(state);

  const auto reachableActions = environment.getReachableActions(state);
  // The difference between this and value evaluation is that we only keep the
  // max value and discard the other actions. We also have no need for the
  // probabilities of taking an action under the policy - as it is assumed to be
  // deterministicly defined by this argmax.
  auto nextValueEstimate = std::accumulate(
      reachableActions.begin(), reachableActions.end(), 0.0f,
      [&](const auto &value, const auto &action) {
        const auto reachableStates =
            environment.getReachableStates(state, action);
        return std::max(value, value_from_state_action(
                                   valueFunction, environment, state, action));
      });

  return nextValueEstimate;
}

/**
 * @brief Perform value iteration to estimate the value function for all states
 * in a valueFunction. Stop when the maximum change in value is less than
 * epsilon.
 */
template <policy::isFiniteStateValueFunction VALUE_FUNCTION_T>
void value_iteration_policy_estimation(
    VALUE_FUNCTION_T &valueFunction,
    const typename VALUE_FUNCTION_T::EnvironmentType &environment,
    const typename VALUE_FUNCTION_T::PrecisionType &epsilon) {

  typename VALUE_FUNCTION_T::PrecisionType delta = 0.0F;
  // sweep over all states and update the value function. When finally no
  // states change significantly we have converged and can exit
  do {
    delta = 0.0F;
    for (const auto &state : environment.getAllPossibleStates()) {
      auto oldValue = valueFunction.valueAt(state);
      auto newValue = value_iteration_policy_estimation_step(
          valueFunction, environment, state);
      delta = std::max(delta, std::abs(oldValue - newValue));
      valueFunction.at(state).value = newValue;
    }
  } while (delta > epsilon);
}

/** @brief Over all states, on a given state output a deterministic policy (an
 * estimate of the optimal) such that the policy simply takes the action with
 * the max value.
 *
 * @details There is no need to look for stability as in the regular case of
 * policy improvement - as we have locked ourselves into a regime where we only
 * want one update. This is because our value estimation is already locked down
 * by its singlular update method and therefore this policy cannot change.
 */
template <policy::isFiniteStateValueFunction VALUE_FUNCTION_T,
          policy::isDistributionPolicy POLICY_T>
void value_iteration(
    VALUE_FUNCTION_T &valueFunction,
    const typename VALUE_FUNCTION_T::EnvironmentType &environment,
    POLICY_T &policy, const typename VALUE_FUNCTION_T::PrecisionType &epsilon) {

  // No loop for policy stability required here as is the case with policy
  // iteration.
  value_iteration_policy_estimation(valueFunction, environment, epsilon);
  policy_improvement(valueFunction, environment, policy);
}

} // namespace markov_decision_process::value_iteration
