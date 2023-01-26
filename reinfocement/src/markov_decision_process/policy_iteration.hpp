#pragma once

#include <unordered_map>

#include "environment.hpp"
#include "markov_decision_process/finite_transition_model.hpp"
#include "policy/distribution_policy.hpp"

namespace dp {

/** @brief V_pi (s): Value function is a funciton that maps states s to their
 * associated values under a given policy pi. The actions are encoded by the
 * policy.
 *
 * @details The value at a given state is the total expected future
 * return (discounted) for future states.
 *
 * where G_t is the return value at t.
 * v_pi (s) = E_pi [ G_t | S_t = s ]
 *          = E_pi [ sum of discounted future returns | S_t = s ] forall s in S
 *          = E_pi [ sum gamma^k * Reward(t+k+1) | S_t = s ] forall s in S
 *
 * When the transition model for the state space is probabilisitc is
 * probabilistic and finite we can describe it with the sum Px * X ...
 *
 * Optimal value functions satasfy the bellman equations
 *
 * v_*(s) = max_(a) E [ Reward_{t+1} + gamma * v_*(S_{t+1}) | S_t=s , A_t=a ]
 * q_*(s,a) = (4.2)
 *
 * where q is the state-action value under * at (s, a)
 *
 */
template <typename POLICY_T, auto INITIAL_VALUE = 0.0F,
          auto DISCOUNT_RATE = 0.0F>
struct ValueFunction {

  using PolicyType = POLICY_T;
  using EnvironmentType = typename POLICY_T::EnvironmentType;
  using StateType = typename EnvironmentType::StateType;
  using TransitionType = typename EnvironmentType::TransitionType;
  using PrecisionType = typename EnvironmentType::PrecisionType;
  using RewardType = typename EnvironmentType::RewardType;

  // The starting value estimate
  constexpr static PrecisionType initial_value = INITIAL_VALUE;
  constexpr static PrecisionType discount_rate = DISCOUNT_RATE;

  std::unordered_map<StateType, PrecisionType, typename StateType::Hash>
      valueEstimates;

  PrecisionType valueAt(const StateType &s) {
    return valueEstimates.emplace(s, initial_value).first->second;
  }

  void initialize(const EnvironmentType &environment) {
    for (const auto &s : environment.getAllPossibleStates()) {
      valueAt(s);
    }
  }
};

template <typename T>
concept isValueFunction = std::is_base_of_v<
    ValueFunction<typename T::PolicyType, T::initial_value, T::discount_rate>,
    T>;

// This mechanism requires the transition model for the finite state
// markov model
template <isValueFunction VALUE_FUNCTION_T,
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

template <isValueFunction VALUE_FUNCTION_T,
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

} // namespace dp
