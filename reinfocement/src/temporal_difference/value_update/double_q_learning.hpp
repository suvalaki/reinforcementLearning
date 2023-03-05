#pragma once
#include <limits>
#include <numeric>
#include <tuple>
#include <utility>

#include "policy/finite/epsilon_greedy_policy.hpp"
#include "policy/objectives/finite_value_function.hpp"
#include "temporal_difference/value_update/td0_updater.hpp"

namespace temporal_difference {

template <policy::objectives::isFiniteStateValueFunction VALUE_FUNCTION_T, class E = xt::random::default_engine_type>
requires policy::objectives::isStateActionKeymaker<typename VALUE_FUNCTION_T::KeyMaker> &&
    policy::isFiniteEpsilonSoftPolicy<VALUE_FUNCTION_T>
struct DoubleQLearningUpdater : TDValueUpdaterBase<DoubleQLearningUpdater<VALUE_FUNCTION_T>, VALUE_FUNCTION_T> {

  // This is SARSA when valueFunction == policy
  SETUP_TYPES_FROM_NESTED_ENVIRON(SINGLE_ARG(VALUE_FUNCTION_T::EnvironmentType));
  using KeyMaker = typename VALUE_FUNCTION_T::KeyMaker;
  using EngineType = E;
  EngineType &engine = xt::random::get_default_random_engine();
  constexpr static PrecisionType QPROB = 0.5F;

  template <policy::isFinitePolicyValueFunctionMixin POLICY_T0, policy::isFinitePolicyValueFunctionMixin POLICY_T1>
  requires std::is_same_v<typename VALUE_FUNCTION_T::KeyType, typename POLICY_T0::KeyType>
      std::tuple<bool, ActionSpace, TransitionType, PrecisionType> double_q_learning_step(
          VALUE_FUNCTION_T &valueFunction, // Should use an additive combination here of policy0 and policy1
          POLICY_T0 &policy0,
          POLICY_T1 &policy1,
          EnvironmentType &environment,
          const ActionSpace &action,
          const PrecisionType &discountRate) {

    // Take Action - according to an epsilon greedy policy.
    const auto nextAction = valueFunction(environment, environment.state);

    // Take Action - By stepping the environment automatically updates the state.
    const auto transition = environment.step(nextAction);
    environment.update(transition);
    const auto reward = RewardType::reward(transition);

    return {transition.isDone(), action, transition, reward};
  }

  void updatePolicy(typename VALUE_FUNCTION_T::EnvironmentType &environment,
                    policy::isFinitePolicyValueFunctionMixin auto &onPolicy,
                    policy::isFinitePolicyValueFunctionMixin auto &offPolicy,
                    const VALUE_FUNCTION_T::KeyType &keyCurrent,
                    const typename VALUE_FUNCTION_T::PrecisionType &reward,
                    const typename VALUE_FUNCTION_T::PrecisionType &discountRate) {

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
  }

  template <policy::isFinitePolicyValueFunctionMixin POLICY_T0, policy::isFinitePolicyValueFunctionMixin POLICY_T1>
  requires std::is_same_v<typename VALUE_FUNCTION_T::KeyType, typename POLICY_T0::KeyType> std::pair<bool, ActionSpace>
  update(VALUE_FUNCTION_T &valueFunction,
         POLICY_T0 &policy0,
         POLICY_T1 &policy1,
         EnvironmentType &environment,
         const ActionSpace &action,
         const PrecisionType &discountRate) {

    const auto [isDone, nextAction, transition, reward] =
        this->double_q_learning_step(valueFunction, policy0, policy1, environment, action, discountRate);

    // with prob 0.5 update q1 using thegreedy action from q2 else update q2 using the greedy action from q1
    const auto key = KeyMaker::make(environment, transition.state, transition.action);
    if (xt::random::rand<double>(xt::xshape<1>{}, 0, 1, this->engine)[0] < this->QPROB) {
      updatePolicy(environment, policy0, policy1, key, reward, discountRate);
    } else {
      updatePolicy(environment, policy0, policy1, key, reward, discountRate);
    }

    return {transition.isDone(), nextAction};
  }
};

} // namespace temporal_difference
