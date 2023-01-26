#pragma once

#include "environment.hpp"
#include "finite_transition_model.hpp"

namespace environment {

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
template <EnvironmentType ENVIRONMENT_T, auto INITIAL_VALUE = 0.0F,
          auto DISCOUNT_RATE = 0.0F>
struct ValueFunction {

  SETUP_TYPES(SINGLE_ARG(ENVIRONMENT_T));
  using EnvironmentType = ENVIRONMENT_T;

  // The starting value estimate
  constexpr static PrecisionType initial_value = INITIAL_VALUE;
  constexpr static PrecisionType discount_rate = DISCOUNT_RATE;

  virtual PrecisionType valueAt(const StateType &s) = 0;
  virtual void initialize(const EnvironmentType &environment) = 0;
};

template <typename T>
concept isValueFunction =
    std::is_base_of_v<ValueFunction<typename T::EnvironmentType,
                                    T::initial_value, T::discount_rate>,
                      T>;

} // namespace environment

namespace markov_decision_process {

template <environment::MarkovDecisionEnvironmentType ENVIRONMENT_T,
          auto INITIAL_VALUE = 0.0F, auto DISCOUNT_RATE = 0.0F>
struct FiniteStateValueFunction
    : public environment::ValueFunction<ENVIRONMENT_T, INITIAL_VALUE,
                                        DISCOUNT_RATE> {

  SETUP_TYPES(SINGLE_ARG(
      environment::ValueFunction<ENVIRONMENT_T, INITIAL_VALUE, DISCOUNT_RATE>));
  using EnvironmentType = ENVIRONMENT_T;

  using BaseType::discount_rate;
  using BaseType::initial_value;

  std::unordered_map<StateType, PrecisionType, typename StateType::Hash>
      valueEstimates;

  PrecisionType valueAt(const StateType &s) override {
    return valueEstimates.emplace(s, initial_value).first->second;
  }

  void initialize(const EnvironmentType &environment) override {
    for (const auto &s : environment.getAllPossibleStates()) {
      valueAt(s);
    }
  }
};

template <typename T>
concept isFiniteStateValueFunction = std::is_base_of_v<
    FiniteStateValueFunction<typename T::EnvironmentType, T::initial_value,
                             T::discount_rate>,
    T>;

} // namespace markov_decision_process