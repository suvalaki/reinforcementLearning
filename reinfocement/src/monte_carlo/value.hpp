#pragma once

#include <unordered_map>
#include <vector>

#include "markov_decision_process/finite_state_value_function.hpp"
#include "monte_carlo/episode.hpp"
#include "policy/distribution_policy.hpp"

namespace monte_carlo {

/// Each occurance of a state within an episode is called a visit.
template <markov_decision_process::isFiniteStateValueFunction VALUE_FUNCTION_T>
using AverageReturnsMap =
    std::unordered_map<typename VALUE_FUNCTION_T::StateType,
                       std::vector<typename VALUE_FUNCTION_T::PrecisionType>,
                       typename VALUE_FUNCTION_T::StateType::Hash>;

template <markov_decision_process::isFiniteStateValueFunction VALUE_FUNCTION_T>
AverageReturnsMap<VALUE_FUNCTION_T> n_visit_returns_initialisation(
    VALUE_FUNCTION_T &valueFunction,
    const typename VALUE_FUNCTION_T::EnvironmentType &environment) {

  // Initialisation step
  // initialise the value function to the initial value
  valueFunction.initialize(environment);
  // initialise returns list for every state
  AverageReturnsMap<VALUE_FUNCTION_T> returns =
      AverageReturnsMap<VALUE_FUNCTION_T>();

  for (const auto &s : environment.getAllPossibleStates()) {
    returns[s] = std::vector<typename VALUE_FUNCTION_T::PrecisionType>();
  }

  return returns;
}

template <environment::EnvironmentType ENVIRON_T> struct StopCondition {
  using EnvironmentType = ENVIRON_T;
  virtual bool operator()(
      std::vector<typename ENVIRON_T::TransitionType>::const_iterator start,
      std::vector<typename ENVIRON_T::TransitionType>::const_iterator end,
      const typename ENVIRON_T::EnvironmentType::TransitionType &transition)
      const = 0;
};

template <typename T>
concept isStopCondition =
    std::is_base_of_v<StopCondition<typename T::EnvironmentType>, T>;

template <environment::EnvironmentType ENVIRON_T>
struct FirstVisitStopCondition : StopCondition<ENVIRON_T> {
  // Only update the average for the first visit in the episode
  bool operator()(
      std::vector<typename ENVIRON_T::TransitionType>::const_iterator start,
      std::vector<typename ENVIRON_T::TransitionType>::const_iterator end,
      const typename ENVIRON_T::EnvironmentType::TransitionType &transition)
      const override {
    // if any of the earlier transitions have the state then they make
    // a better first visit.
    return not std::any_of(start, end, [&transition](const auto &t) {
      return t.state == transition.state;
    });
  }
};

template <std::size_t max_episode_length,
          markov_decision_process::isFiniteStateValueFunction VALUE_FUNCTION_T,
          policy::isDistributionPolicy POLICY_T,
          isStopCondition STOP_CONDITION_T>
void visit_valueEstimate_step(
    VALUE_FUNCTION_T &valueFunction,
    typename VALUE_FUNCTION_T::EnvironmentType &environment, POLICY_T &policy,
    AverageReturnsMap<VALUE_FUNCTION_T> &returns,
    const STOP_CONDITION_T &stop_condition) {

  using EnvironmentType = typename VALUE_FUNCTION_T::EnvironmentType;

  // Generate an episode following policy pi
  Episode<EnvironmentType> episode;
  if constexpr (max_episode_length == 0) {
    // It is assumed that the terminal state will be generated in the normal
    // operation of the environments step funciton.
    episode = generate_episode(environment, policy);
  } else {
    // Either the step function will generate an episode termination or the
    // maximum episode length will be reached - and so the episode will stop
    // early.
    episode = generate_episode(environment, policy, max_episode_length);
  }

  // Initialise the return to 0
  typename VALUE_FUNCTION_T::PrecisionType G = 0;
  // Loop over the transitions in the episode in reverse order
  for (auto it = episode.GetTransitions().rbegin();
       it != episode.GetTransitions().rend(); ++it) {
    // Update the return
    G = EnvironmentType::RewardType::reward(*it) +
        valueFunction.discount_rate * G;

    // If the state does not appear in an earlier transition
    if (stop_condition(episode.GetTransitions().begin(), (it + 1).base(),
                       *it)) {
      // Add the return to the list of returns
      returns[it->state].push_back(G);
      // Update the value function to be the average of returns from that
      // state
      valueFunction.valueEstimates[it->state] =
          std::accumulate(returns[it->state].begin(), returns[it->state].end(),
                          0.0F) /
          returns[it->state].size();
    }
  }
}

/** @brief The value v_{pi}(s) is the average of all returns following the first
 * visit to s. */
template <std::size_t max_episode_length,
          markov_decision_process::isFiniteStateValueFunction VALUE_FUNCTION_T,
          policy::isDistributionPolicy POLICY_T>
void first_visit_valueEstimate(
    VALUE_FUNCTION_T &valueFunction,
    typename VALUE_FUNCTION_T::EnvironmentType &environment, POLICY_T &policy,
    std::size_t episodes) {

  // initialise the value function and returns list for every state
  auto returns = n_visit_returns_initialisation(valueFunction, environment);

  // Loop forever over episodes - Here we actually only loop for the requested
  // number of episodes
  for (std::size_t i = 0; i < episodes; ++i) {
    visit_valueEstimate_step<max_episode_length>(
        valueFunction, environment, policy, returns,
        FirstVisitStopCondition<typename VALUE_FUNCTION_T::EnvironmentType>());
  }
}

template <environment::EnvironmentType ENVIRON_T>
struct EveryVisitStopCondition : StopCondition<ENVIRON_T> {
  // We always update the average at every visit.
  bool operator()(
      std::vector<typename ENVIRON_T::TransitionType>::const_iterator start,
      std::vector<typename ENVIRON_T::TransitionType>::const_iterator end,
      const typename ENVIRON_T::TransitionType &transition) const override {
    return true;
  }
};

/** @brief The value v_{pi}(s) is the average of all returns following all
 * visits to s. */
template <std::size_t max_episode_length,
          markov_decision_process::isFiniteStateValueFunction VALUE_FUNCTION_T,
          policy::isDistributionPolicy POLICY_T>
void every_visit_valueEstimate(
    VALUE_FUNCTION_T &valueFunction,
    typename VALUE_FUNCTION_T::EnvironmentType &environment, POLICY_T &policy,
    std::size_t episodes) {

  // initialise the value function and returns list for every state
  auto returns = n_visit_returns_initialisation(valueFunction, environment);

  // Loop forever over episodes - Here we actually only loop for the requested
  // number of episodes
  for (std::size_t i = 0; i < episodes; ++i) {
    visit_valueEstimate_step<max_episode_length>(
        valueFunction, environment, policy, returns,
        EveryVisitStopCondition<typename VALUE_FUNCTION_T::EnvironmentType>());
  }
}

} // namespace monte_carlo
