#pragma once

#include <unordered_map>

#include "reinforce/environment.hpp"
#include "reinforce/markov_decision_process/finite_transition_model.hpp"
#include "reinforce/policy/distribution_policy.hpp"
#include "reinforce/policy/objectives/finite_value_function.hpp"
#include "reinforce/policy/value.hpp"

// The key concept for MDPs is that the best policy can always be determined
// by looking at the value for each state. This is because whenever we find
// ourselves at a state we can always choose the action that maximises the
// future state return. We can always know which deterministic policy to
// take that will maximise future returns as we know the transition model
// already. This is the basis of policy iteration. We are estimating V(S).
// Given the transition model and V(S)then q(s,a) (the value of taking action a
// in state s) is KNOWN.
namespace markov_decision_process {

/**
 * @brief The exected future Value funciton for a given state under a
 * determined action.
 *
 * @details Under a markov decision process the expected future value of a
 * state is the sum of the expected reward for each reachable state and the
 * expected return of that state following the action. This is given by the
 * following \begin{equation} E_{pi} [R_{t+1} + gamma * V_{pi}(S_{t+1}) |
 * S_{t} = s, A_{t} = a] = \sum_{s', r} p(s', r | s, a) [r + gamma *
 * V_{pi}(s')] \end{equation} The Action is fixed in this situation.
 *
 * @tparam VALUE_FUNCTION_T The value function type which will hold a
 * mapping from state to value
 * @param valueFunction The value function to use for the value estimates
 * @param environment The environment to use for the transition model
 * @param state The state to evaluate the value for
 * @param action The action to evaluate the value for. Alongside state and
 * the transition model this defines the reachable states
 * @return VALUE_FUNCTION_T::PrecisionType The expected value
 */
template <policy::objectives::isFiniteStateValueFunction VALUE_FUNCTION_T>
typename VALUE_FUNCTION_T::PrecisionType value_from_state_action(
    VALUE_FUNCTION_T &valueFunction,
    const typename VALUE_FUNCTION_T::EnvironmentType &environment,
    const typename VALUE_FUNCTION_T::EnvironmentType::StateType &state,
    const typename VALUE_FUNCTION_T::EnvironmentType::ActionSpace &action) {

  SETUP_TYPES_W_VALUE_FUNCTION(VALUE_FUNCTION_T);

  const auto &transitionModel = environment.transitionModel;

  auto reachableStates = environment.getReachableStates(state, action);
  return std::accumulate(
      reachableStates.begin(), reachableStates.end(), 0.0F, [&](const auto &value, const auto &nextState) {
        auto transition = TransitionType{state, action, nextState};

        if (transitionModel.transitions.find(transition) == transitionModel.transitions.end())
          return value;

        return value + transitionModel.transitions.at(transition) *
                           (RewardType::reward(transition) +
                            valueFunction.discount_rate *
                                valueFunction.emplace(nextState, valueFunction.initial_value).first->second.value);
      });
}

// This mechanism requires the transition model for the finite state
// markov model

/**
 * @brief Perform a single step of policy evaluation for a given state.
 *
 * @details This is the core of the policy evaluation for a single state.
 * From this state what is the total expected future Return given we start in
 * state S. This is given by the Return of the state action pair and the
 * expected return of the next state reached by taking those actions. The policy
 * implies the probability of taking each action from this state. Hence the
 * value function under policy
 * $\pi$ at S is given by:
 *
 * \begin{equation}
 * v_{\pi}(s) = E_{\pi} [G_{t} | S_{t} = s]
 * = E_{\pi} [R_{t+1} + gamma * G_{t+1} | S_{t} = s]
 * = E_{\pi} [R_{t+1} + gamma * V_{\pi}(S_{t+1}) | S_{t} = s]
 * = \sum_{a} \pi(a | s) \sum_{s', r} p(s', r | s, a) [r + gamma * v_{\pi}(s')]
 * \end{equation}
 *
 * @tparam VALUE_FUNCTION_T The value function type which will hold a mapping
 * from state to value
 * @tparam POLICY_T The policy type which stochastically decides actions from
 * the given state
 * @param valueFunction The value function to use for the value estimates
 * @param environment The environment to use for the transition model
 * @param policy The policy to use for the action selection
 * @param state The state to evaluate the value for
 * @return VALUE_FUNCTION_T::PrecisionType The estimated value of the state
 */
template <policy::objectives::isFiniteStateValueFunction VALUE_FUNCTION_T, policy::isDistributionPolicy POLICY_T>
typename VALUE_FUNCTION_T::PrecisionType policy_evaluation_step(
    VALUE_FUNCTION_T &valueFunction,
    const typename VALUE_FUNCTION_T::EnvironmentType &environment,
    const POLICY_T &policy,
    const typename VALUE_FUNCTION_T::StateType &state) {

  const auto &transitionModel = environment.transitionModel;
  auto currentValueEstimate = valueFunction.valueAt(state);

  // For each state action pair reachable from this state evaluate the
  // expected value of the next state given the policy.
  const auto reachableActions = environment.getReachableActions(state);
  auto nextValueEstimate = std::accumulate(
      reachableActions.begin(), reachableActions.end(), 0.0F, [&](const auto &value, const auto &action) {
        const auto reachableStates = environment.getReachableStates(state, action);
        return value + policy.getProbability(environment, state, action) *
                           value_from_state_action(valueFunction, environment, state, action);
      });

  return nextValueEstimate;
}

/**
 * @brief Perform a policy evaluation sweep over all states in the environment.
 *
 * @details This is the core of the policy evaluation. We sweep over all states
 * and perform a step of policy evaluation for each state. When finally no
 * states change significantly we have converged and can exit. The updates occur
 * in place such that the values for some of the updates are actually the
 * updated values from the current itteration - which has been shown to also
 * converge (and converges faster).
 *
 * @tparam VALUE_FUNCTION_T
 * @tparam POLICY_T
 * @param valueFunction The value function to use for the value estimates
 * @param environment The environment to use for the transition model
 * @param policy The policy to use for the action selection
 * @param epsilon The convergence threshold. When the value function at any
 * state changes by less than epsilon we have converged.
 */
template <policy::objectives::isFiniteStateValueFunction VALUE_FUNCTION_T, policy::isDistributionPolicy POLICY_T>
void policy_evaluation(
    VALUE_FUNCTION_T &valueFunction,
    const typename POLICY_T::EnvironmentType &environment,
    POLICY_T &policy,
    const typename VALUE_FUNCTION_T::PrecisionType &epsilon) {

  assert(epsilon > 0.0F);

  typename VALUE_FUNCTION_T::PrecisionType delta = 0.0F;
  // sweep over all states and update the value function. When finally no
  // states change significantly we have converged and can exit
  do {
    delta = 0.0F;
    for (const auto &state : environment.getAllPossibleStates()) {
      auto oldValue = valueFunction.valueAt(state);
      auto newValue = policy_evaluation_step(valueFunction, environment, policy, state);
      delta = std::max(delta, std::abs(oldValue - newValue));
      valueFunction.at(state).value = newValue;
    }
  } while (delta > epsilon and delta > 0.0F);
}

/**
 * @brief Perform a single step of policy improvement. Improve the choice of
 * action taken under the policy from this given state.
 *
 * @details The idea behing policy improvement is to consider selecting action a
 * at state S and thereafter continuing to follow the policy. The value of
 * behaving in this way is the reward under that action plus the value of the
 * next state under the policy.
 *
 * \begin{equation}
 * q_{\pi} (s,a) = E_{\pi} [R_{t+1} + gamma * V_{\pi}(S_{t+1}) | S_{t} = s,
 * A_{t}
 * \end{equation}
 *
 * If this improves the value then taking the new action is a better alternative
 * and is a more optimal policy. Hence we can improve the policy by selecting
 * the action which maximises this value. The policy improvement step is given
 * by:
 *
 * \begin{equation}
 * \pi'(s) = argmax_{a} q_{\pi}(s, a) = argmax_{a} E_{\pi}
 * [R_{t+1} + gamma * V_{\pi}(S_{t+1}) | S_{t} = s, A_{t} = a] = argmax_{a}
 * \sum_{s', r} p(s', r | s, a) [r + gamma * v_{\pi}(s')]
 * \end{equation}
 *
 * @tparam VALUE_FUNCTION_T
 * @tparam POLICY_T
 * @param valueFunction The value function to use for the value estimates
 * @param environment The environment to use for the transition model
 * @param policy The policy to use for the action selection
 * @param state The state to improve the policy for
 * @return true If the policy was improved
 * @return false If the policy was not improved
 */
template <policy::objectives::isFiniteStateValueFunction VALUE_FUNCTION_T, policy::isDistributionPolicy POLICY_T>
bool policy_improvement_step(
    VALUE_FUNCTION_T &valueFunction,
    const typename VALUE_FUNCTION_T::EnvironmentType &environment,
    POLICY_T &policy,
    const typename VALUE_FUNCTION_T::EnvironmentType::StateType &state) {

  using PolicyKeyMaker = typename POLICY_T::KeyMaker;

  // using KeyMaker = typename POLICY_T::KeyMaker;

  const auto oldActions = policy.getProbabilities(environment, state);
  const auto oldActionIdx = std::max_element(
      oldActions.begin(), oldActions.end(), [&](const auto &lhs, const auto &rhs) { return lhs.second < rhs.second; });
  const auto oldAction = PolicyKeyMaker::get_action_from_key(environment, oldActionIdx->first);

  // For each state action pair find the new distribution of actions
  // under the value function. Take all actions with the argmax and set them
  // to an equal and max probability. When this is a single action the
  // update is effectively deterministic and the probability for taking
  // the action = 1.0

  const auto reachableActions = environment.getReachableActions(state);
  auto nextActionIdx =
      std::max_element(reachableActions.begin(), reachableActions.end(), [&](const auto &lhs, const auto &rhs) {
        return value_from_state_action(valueFunction, environment, state, lhs) <
               value_from_state_action(valueFunction, environment, state, rhs);
      });
  const auto nextAction = *nextActionIdx;
  const auto policyStable = oldAction == nextAction;

  // update the policy - by setting the policy to be deterministic on the
  // new argmax
  // warn: under this current approach we always pick a single action
  policy.setDeterministicPolicy(environment, state, nextAction);

  return policyStable;
}

/**
 * @brief Perform policy improvement over all states until the policy is stable
 *
 * @details Policy improvement is performed by considering the value of taking a
 * given action at a given state and then continuing to follow the policy. The
 * value of behaving in this way is the reward under that action plus the value
 * of the next state under the policy.
 *
 * @tparam VALUE_FUNCTION_T
 * @tparam POLICY_T
 * @param valueFunction The value function to use for the value estimates
 * @param environment The environment to use for the transition model
 * @param policy The policy to use for the action selection
 * @return true If the policy is stable (no action updates were made)
 * @return false If the policy is not stable (at least one action update was
 * made)
 */
template <policy::objectives::isFiniteStateValueFunction VALUE_FUNCTION_T, policy::isDistributionPolicy POLICY_T>
bool policy_improvement(
    VALUE_FUNCTION_T &valueFunction, const typename VALUE_FUNCTION_T::EnvironmentType &environment, POLICY_T &policy) {

  bool policyStable = true;

  const auto allStates = environment.getAllPossibleStates();
  for (const auto &state : allStates) {
    policyStable &= policy_improvement_step(valueFunction, environment, policy, state);
  }

  return policyStable;
}

/**
 * @brief Perform policy iteration on the given value function and policy.
 *
 * @details Policy iteration is performed by first performing policy evaluation
 * and then policy improvement.
 *
 * @tparam VALUE_FUNCTION_T
 * @tparam POLICY_T
 * @param valueFunction The value function to use for the value estimates
 * @param environment The environment to use for the transition model
 * @param policy The policy to use for the action selection
 * @param epsilon The precision to use for the policy evaluation
 */
template <policy::objectives::isFiniteStateValueFunction VALUE_FUNCTION_T, policy::isDistributionPolicy POLICY_T>
void policy_iteration(
    VALUE_FUNCTION_T &valueFunction,
    const typename VALUE_FUNCTION_T::EnvironmentType &environment,
    POLICY_T &policy,
    const typename VALUE_FUNCTION_T::PrecisionType &epsilon) {

  bool policyStable = true;

  // While any of the policies are not stable keep improving them
  do {
    policy_evaluation(valueFunction, environment, policy, epsilon);
    policyStable &= policy_improvement(valueFunction, environment, policy);

  } while (not policyStable);
}

} // namespace markov_decision_process
