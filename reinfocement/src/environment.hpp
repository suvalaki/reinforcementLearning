#pragma once

#include <cmath>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <utility>
#include <variant>

#include <array>

#include "action.hpp"



template <typename T>
concept Float = std::is_floating_point<T>::value;

namespace environment {

template <Float TYPE_T> struct State { using PrecisionType = TYPE_T; };

template <typename STATE_T>
concept StateType =
    std::is_base_of_v<State<typename STATE_T::PrecisionType>, STATE_T>;

template <StateType STATE_T, action::CompositeArraySpecType T> struct Action : T::DataType {
  using SpecType = T;
  using StateType = STATE_T;
  using PrecisionType = typename StateType::PrecisionType;

  // Add type which adheres to the spec
  Action(const typename T::DataType& d) : T::DataType(d) {}

  // When an action modifies the state space impliment anew
  virtual StateType step(const StateType &state) const { return state; }
};

template <typename ACTION_T>
concept ActionType = std::is_base_of_v<
    Action<typename ACTION_T::StateType, typename ACTION_T::SpecType>,
    ACTION_T>;

// TODO : CONCEPT TO REQUIRE ALL ACTIONS TO BE OVER THE SAME BASE

template <typename... T> struct Step {};
template <ActionType ACTION0> struct Step<ACTION0> {
  using StateType = typename ACTION0::StateType;
  using PrecisionType = typename StateType::PrecisionType;
  using ActionSpace = ACTION0;
  using SpecType = typename ActionSpace::SpecType;

  // Calculate the next state given the current state and the action
  // For more complex models like markov transition states this can be overriden
  static StateType step(const StateType &state, const ActionSpace &action) {
    return action.step(state);
  };
};

template <typename STEP_T>
concept StepType =
    std::is_base_of_v<Step<typename STEP_T::ActionSpace>, STEP_T>;

template <typename... T> struct Transition {};
template <ActionType ACTION0> struct Transition<ACTION0> {
  using StateType = typename ACTION0::StateType;
  using PrecisionType = typename StateType::PrecisionType;
  using ActionSpace = ACTION0;
  using ActionSpecType = typename ActionSpace::SpecType;
  using StepType = Step<ActionSpace>;

  StateType state;
  ActionSpace action;
  StateType nextState;
};

template <std::size_t SEQUENCE_LENGTH, typename... T>
struct TransitionSequence {};
template <std::size_t SEQUENCE_LENGTH, ActionType ACTION0>
struct TransitionSequence<SEQUENCE_LENGTH, ACTION0>
    : std::array<Transition<ACTION0>, SEQUENCE_LENGTH> {
  using StateType = typename ACTION0::StateType;
  using PrecisionType = typename StateType::PrecisionType;
  using ActionSpace = ACTION0;
  using ActionSpecType = typename ActionSpace::SpecType;
  using StepType = Step<ActionSpace>;
  using TransitionType = Transition<ActionSpace>;
  constexpr static std::size_t LENGTH = SEQUENCE_LENGTH;
};

template <typename... T> struct Reward {};
template <ActionType ACTION0> struct Reward<ACTION0> {
  using StateType = typename ACTION0::StateType;
  using PrecisionType = typename StateType::PrecisionType;
  using ActionSpace = ACTION0;
  using ActionSpecType = typename ActionSpace::SpecType;
  using StepType = Step<ActionSpace>;
  using TransitionType = Transition<ActionSpace>;
  static PrecisionType reward(const TransitionType &t) { return 0.0F; }
};

template <typename REWARD_T>
concept RewardType =
    std::is_base_of_v<Reward<typename REWARD_T::ActionSpace>, REWARD_T>;

template <typename REWARD_T>
concept RewardProtocol = requires(REWARD_T t,
                                  const typename REWARD_T::TransitionType &tn) {
  // Reqyure that the Object implementing the reward protocol to implement
  // static PrecisionType reward(const TransitionType& t) {return 0.0F;}
  {REWARD_T::reward(tn)};
  // Require Precision Type is inside REWARD_D
  typename REWARD_T::PrecisionType;
  // Require ActionSpace is inside REWARD_D - Rewards always act over an action
  // space
  typename REWARD_T::ActionSpace;
};

template <RewardProtocol REWARD_T> struct Return {

  using PrecisionType = REWARD_T::PrecisionType;
  using RewardType = REWARD_T;
  using ActionSpace = typename RewardType::ActionSpace;
  using ActionSpecType = typename ActionSpace::SpecType;

  // Discounted Return of future SEQUENCE_LENGTH time steps
  template <std::size_t SEQUENCE_LENGTH>
  static PrecisionType
  value(const TransitionSequence<SEQUENCE_LENGTH,
                                 typename RewardType::ActionSpace> &t) {
    auto arr = future_value(t);
    return std::accumulate(arr.begin(), arr.end(),
                           static_cast<PrecisionType>(0));
  }

  // Return raw rewards for future time steps without discounting or any other
  // calcs
  template <std::size_t SEQUENCE_LENGTH>
  static std::array<PrecisionType, SEQUENCE_LENGTH>
  future_value(const TransitionSequence<SEQUENCE_LENGTH, ActionSpace> &t) {
    std::array<PrecisionType, SEQUENCE_LENGTH> result;
    for (int i = 0; i < SEQUENCE_LENGTH; i++) {
      result[i] = RewardType::reward(t[i]);
    }
    return result;
  }
};

template <typename RETURN_T>
concept ReturnType =
    std::is_base_of_v<Return<typename RETURN_T::RewardType>, RETURN_T>;

template <RewardProtocol REWARD_T, float DISCOUNT = 0.8F>
struct DiscountReturn : Return<REWARD_T> {

  using BaseType = Return<REWARD_T>;
  constexpr static REWARD_T::PrecisionType discountRate = DISCOUNT;

  template <std::size_t SEQUENCE_LENGTH>
  static BaseType::PrecisionType value(
      const TransitionSequence<SEQUENCE_LENGTH,
                               typename BaseType::RewardType::ActionSpace> &t) {
    auto arr = BaseType::future_value(t);
    typename BaseType::PrecisionType result = 0.0F;
    for (int i = 0; i < SEQUENCE_LENGTH; i++) {
      result += arr[i] * static_cast<typename BaseType::PrecisionType>(
                             std::pow(discountRate, i));
    }
    return result;
  }
};

template <StepType STEP_T, RewardType REWARD_T, ReturnType RETURN_T>
struct Environment {

  using StateType = typename STEP_T::StateType;
  using ActionSpace = typename STEP_T::ActionSpace;
  using ActionSpecType = typename ActionSpace::SpecType;
  using PrecisionType = typename STEP_T::PrecisionType;
  using StepType = STEP_T;
  using TransitionType = Transition<ActionSpace>;
  using RewardType = REWARD_T;
  using ReturnType = RETURN_T;

  template <std::size_t EPISODE_LENGTH>
  using EpisodeType = TransitionSequence<EPISODE_LENGTH, ActionSpace>;

  StateType state;

  Environment() = default;
  Environment(const StateType &s) : state(s) {}

  virtual void reset() = 0;
  virtual TransitionType step(const ActionSpace &action) {
    return TransitionType{state, action, StepType::step(state, action)};
  };
  void update(const TransitionType &t) { state = t.nextState; }
};

template <typename ENVIRON_T>
concept EnvironmentType = std::is_base_of_v<
    Environment<typename ENVIRON_T::StepType, typename ENVIRON_T::RewardType,
                typename ENVIRON_T::ReturnType>,
    ENVIRON_T>;

} // namespace environment
