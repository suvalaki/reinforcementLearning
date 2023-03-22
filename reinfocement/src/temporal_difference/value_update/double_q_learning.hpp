#pragma once
#include <limits>
#include <numeric>
#include <tuple>
#include <utility>

#include "policy/finite/epsilon_greedy_policy.hpp"
#include "policy/objectives/finite_value_function.hpp"
#include "temporal_difference/value_update/value_update.hpp"

#define DQU DoubleQLearningUpdater<VALUE_FUNCTION_T, E>

namespace temporal_difference {

template <typename CRTP>
struct DoubleQLearningStepMixin {

  SETUP_TYPES_FROM_NESTED_ENVIRON(SINGLE_ARG(CRTP::EnvironmentType));

  /**
   * @brief Prior to each update of the policies (either Q1 or Q2) in double Q learning we select an action A from a
   * GreedyEpsilon policy (where the values are the sum Q1+Q2) and then find the assiciated reward by querying the
   * environment under A. We say that the environment moves from S under A to S' and the reward is R.
   * @return std::tuple<bool, ActionSpace, TransitionType, PrecisionType>
   */
  template <policy::isFinitePolicyValueFunctionMixin POLICY_T0, policy::isFinitePolicyValueFunctionMixin POLICY_T1>
  requires std::is_same_v<typename CRTP::KeyType, typename POLICY_T0::KeyType>
  auto step(
      typename CRTP::ValueFunctionType &valueFunction,
      POLICY_T0 &policy0,
      POLICY_T1 &policy1,
      EnvironmentType &environment,
      const ActionSpace &action,
      const PrecisionType &discountRate) -> typename CRTP::StatefulUpdateResult {

    // Take Action - according to an epsilon greedy policy. ON THE VALUE FUNCTION
    const auto nextAction = valueFunction(environment, environment.state);

    // Take Action - By stepping the environment automatically updates the state.
    // Note that id doesnt really matter when the action is taken on the environment as long as we keep track
    // of which state we were on to perform out update.
    const auto transition = environment.step(nextAction);
    environment.update(transition);
    const auto reward = RewardType::reward(transition);

    return {transition.isDone(), action, transition, reward};
  }
};

template <typename CRTP, class E = xt::random::default_engine_type>
struct DoubleQLearningValueUpdateMixin {

  SETUP_TYPES_FROM_NESTED_ENVIRON(SINGLE_ARG(CRTP::EnvironmentType));
  using KeyMaker = typename CRTP::KeyMaker;
  using KeyType = typename CRTP::KeyType;
  using EngineType = E;

  EngineType &engine = xt::random::get_default_random_engine();
  constexpr static PrecisionType QPROB = 0.5F;

  /** @brief Update one of the policies (Q1 or Q2) in double Q learning. Generalised
   * by swapping which Q* is onPolicy vs offPolicy. This updates the OFF POLICY
   *
   * @details The update for double Q learning is as follows:
   *  Let Q1 be the on policy value function and Q2 be the off policy value function.
   *  Q1[s,a] <- Q1[s,a] + alpha * (r + gamma * Q2[s',argmax_a(Q1[s',a])] - Q1[s,a])
   */
  void updatePolicy(
      EnvironmentType &environment,
      policy::isFinitePolicyValueFunctionMixin auto &onPolicy,
      policy::isFinitePolicyValueFunctionMixin auto &offPolicy,
      const KeyType &keyCurrent,
      const PrecisionType &reward,
      const PrecisionType &discountRate) {

    // get the max value from the next state.
    // This is the difference between SARSA and Q-Learning.
    const auto reachableActions = environment.getReachableActions(environment.state);

    // Get Argmax Action from off policy
    const auto onPolicyArgmaxAction = offPolicy.getArgmaxAction(environment, environment.state);
    const auto offPolicyValueOfOnPolicyAction =
        offPolicy.valueAt(KeyMaker::make(environment, environment.state, onPolicyArgmaxAction));
    auto &offPolicyValueOfOffPolicyAction = offPolicy[keyCurrent].value;

    offPolicyValueOfOffPolicyAction =
        offPolicyValueOfOffPolicyAction +
        temporal_differenc_error(offPolicyValueOfOffPolicyAction, offPolicyValueOfOnPolicyAction, reward, discountRate);
  };

  /**
   * @brief Complete control flow for double Q Learning. An ubiased version of Q learning.
   *
   * @details In double Q learning we seek to eliminate the source of positive bias in Q learning. The bias comes from
   * the fact that the policy is used to select the next action.
   * In Q learning we select the value of the argmax action from the policy in the new state as an estimate of the
   * value of the next state-action value. When the policy is updated with the current value function, the policy
   * selects its own argmax and so is biased towards itself. This policy could be wrong though. In order to eliminate
   * this bias we introduce two policies Q1 and Q2 and a value function V which is the sum of Q1 + Q2. We can achieve
   * an unbiased estimate to sample from (unbiased to policy Q1 versus Q2) by sampling actions from V to direct the
   * environment; and the next state where we will evaluate the policy of. At that stage we can randomly select to
   * update (i) Q1 using the the action specified by Q1 with the temporal difference error under Q2 or update (ii) Q2
   * using the action specified by Q2 with the temporal difference error under Q1. Using each policy to control the
   * other.
   *
   * ```
   * for each step in epsiode:
   *  s <- current state
   *  a <- action from policy V = Q1 + Q2 chosen via an EpslonSoft Policy
   *  s' <- next state from environment by taking action a from s
   *  r <- reward
   *  choose Q1 or Q2 with prob 0.5:
   *    if Q1:
   *      Q1[s,a] <- Q1[s,a] + alpha * (r + gamma * Q2[s',argmax_a(Q1[s',a])] - Q1[s,a])
   *    else:
   *      Q2[s,a] <- Q2[s,a] + alpha * (r + gamma * Q1[s',argmax_a(Q2[s',a])] - Q2[s,a])
   *  move to next state:
   *  current state <- s'
   * ```
   *
   * Double Q Learning was proposed in:
   * https://proceedings.neurips.cc/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html
   *
   *
   * @return std::pair<bool, ActionSpace>
   */
  template <policy::isFinitePolicyValueFunctionMixin POLICY_T0, policy::isFinitePolicyValueFunctionMixin POLICY_T1>
  requires std::is_same_v<typename CRTP::ValueFunctionType ::KeyType, typename POLICY_T0::KeyType>
  auto updateValue(
      typename CRTP::ValueFunctionType &valueFunction,
      POLICY_T0 &policy0,
      POLICY_T1 &policy1,
      EnvironmentType &environment,
      const KeyType &key,
      const KeyType &keyNext,
      const PrecisionType &reward,
      const PrecisionType &discountRate) -> void {

    // with prob 0.5 update q1 using thegreedy action from q2 else update q2 using the greedy action from q1
    if (xt::random::rand<double>(xt::xshape<1>{}, 0, 1, this->engine)[0] < this->QPROB) {
      updatePolicy(environment, policy0, policy1, key, reward, discountRate);
    } else {
      updatePolicy(environment, policy1, policy0, key, reward, discountRate);
    }
  }
};

template <policy::objectives::isFiniteAdditiveValueFunctionCombination V, class E = xt::random::default_engine_type>
requires policy::objectives::isStateActionKeymaker<typename V::KeyMaker> && policy::isFiniteEpsilonSoftPolicy<V>
using DoubleQLearningUpdater =
    TemporalDifferenceValueUpdater<V, DoubleQLearningStepMixin, DoubleQLearningValueUpdateMixin>;

} // namespace temporal_difference
