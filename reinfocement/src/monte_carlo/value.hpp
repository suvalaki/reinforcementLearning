#pragma once

#include <unordered_map>
#include <vector>

#include "monte_carlo/episode.hpp"
#include "policy/distribution_policy.hpp"
#include "policy/value.hpp"

namespace monte_carlo {

// The SIMPLEST monte carlo control takes a Key over the state types. But this may not be best when
// we dont know the state dynamics.

/// Each occurance of a state within an episode is called a visit.
template <policy::isFiniteStateValueFunction VALUE_FUNCTION_T>
using AverageReturnsMap = std::unordered_map<typename VALUE_FUNCTION_T::KeyType,
                                             std::vector<typename VALUE_FUNCTION_T::PrecisionType>,
                                             typename VALUE_FUNCTION_T::KeyMaker::Hash>;

template <policy::isStateActionValueFunction VALUE_FUNCTION_T>
AverageReturnsMap<VALUE_FUNCTION_T>
n_visit_returns_initialisation(VALUE_FUNCTION_T &valueFunction,
                               typename VALUE_FUNCTION_T::EnvironmentType &environment) {

  // Initialisation step
  // initialise the value function to the initial value
  valueFunction.initialize(environment);
  // initialise returns list for every state
  AverageReturnsMap<VALUE_FUNCTION_T> returns = AverageReturnsMap<VALUE_FUNCTION_T>();

  for (const auto &s : environment.getAllPossibleStates()) {
    for (const auto &a : environment.getReachableActions(s)) {
      returns[{s, a}] = std::vector<typename VALUE_FUNCTION_T::PrecisionType>();
    }
  }

  return returns;
}

template <policy::isStateValueFunction VALUE_FUNCTION_T>
AverageReturnsMap<VALUE_FUNCTION_T>
n_visit_returns_initialisation(VALUE_FUNCTION_T &valueFunction,
                               typename VALUE_FUNCTION_T::EnvironmentType &environment) {

  // Initialisation step
  // initialise the value function to the initial value
  valueFunction.initialize(environment);
  // initialise returns list for every state
  AverageReturnsMap<VALUE_FUNCTION_T> returns = AverageReturnsMap<VALUE_FUNCTION_T>();

  for (const auto &s : environment.getAllPossibleStates()) {
    returns[s] = std::vector<typename VALUE_FUNCTION_T::PrecisionType>();
  }

  return returns;
}

template <policy::isActionValueFunction VALUE_FUNCTION_T>
AverageReturnsMap<VALUE_FUNCTION_T>
n_visit_returns_initialisation(VALUE_FUNCTION_T &valueFunction,
                               typename VALUE_FUNCTION_T::EnvironmentType &environment) {

  // Initialisation step
  // initialise the value function to the initial value
  valueFunction.initialize(environment);
  // initialise returns list for every state
  AverageReturnsMap<VALUE_FUNCTION_T> returns = AverageReturnsMap<VALUE_FUNCTION_T>();

  for (const auto &a : environment.getAllPossibleActions()) {
    returns[a] = std::vector<typename VALUE_FUNCTION_T::PrecisionType>();
  }

  return returns;
}

template <policy::isNotKnownValueFunction VALUE_FUNCTION_T>
AverageReturnsMap<VALUE_FUNCTION_T>
n_visit_returns_initialisation(VALUE_FUNCTION_T &valueFunction,
                               typename VALUE_FUNCTION_T::EnvironmentType &environment) {

  // Initialisation step
  // initialise the value function to the initial value
  valueFunction.initialize(environment);
  // initialise returns list for every state
  AverageReturnsMap<VALUE_FUNCTION_T> returns = AverageReturnsMap<VALUE_FUNCTION_T>();

  for (const auto &s : environment.getAllPossibleStates()) {
    for (const auto &a : environment.getReachableActions(s)) {
      returns[typename VALUE_FUNCTION_T::KeyMaker::make(s, a)] =
          std::vector<typename VALUE_FUNCTION_T::PrecisionType>();
    }
  }

  return returns;
}

template <policy::isFiniteStateValueFunction VALUE_FUNCTION_T> struct StopCondition {
  SETUP_TYPES_FROM_ENVIRON(SINGLE_ARG(VALUE_FUNCTION_T::EnvironmentType));
  using ValueFunctionType = VALUE_FUNCTION_T;

  // template <typename EPISODE_T>
  // bool operator()(typename EPISODE_T::DataContainer::const_iterator start,
  //                 typename EPISODE_T::DataContainer::const_iterator end,
  //                 const typename ENVIRON_T::EnvironmentType::TransitionType
  //                     &transition) const = 0;
};

template <typename T>
concept isStopCondition = std::is_base_of_v<StopCondition<typename T::ValueFunctionType>, T>;

template <policy::isFiniteStateValueFunction VALUE_FUNCTION_T>
struct FirstVisitStopCondition : StopCondition<VALUE_FUNCTION_T> {

  using KeyMaker = VALUE_FUNCTION_T::KeyMaker;

  // Only update the average for the first visit in the episode
  template <typename EPISODE_T>
  bool operator()(typename EPISODE_T::DataContainer::const_iterator start,
                  typename EPISODE_T::DataContainer::const_iterator end,
                  const typename VALUE_FUNCTION_T::EnvironmentType::TransitionType &transition) const {
    // if any of the earlier transitions have the key then they make
    // a better first visit.
    return not std::any_of(start, end, [&transition](const auto &t) {
      return KeyMaker::make(t.state, t.action) == KeyMaker::make(transition.state, transition.action);
    });
  }
};

template <std::size_t max_episode_length,
          policy::isFiniteStateValueFunction VALUE_FUNCTION_T,
          policy::isDistributionPolicy POLICY_T,
          isStopCondition STOP_CONDITION_T>
void visit_valueEstimate_step(VALUE_FUNCTION_T &valueFunction,
                              typename VALUE_FUNCTION_T::EnvironmentType &environment,
                              POLICY_T &policy,
                              AverageReturnsMap<VALUE_FUNCTION_T> &returns,
                              const STOP_CONDITION_T &stop_condition) {

  using EnvironmentType = typename VALUE_FUNCTION_T::EnvironmentType;
  using EpisodeType = std::
      conditional_t<max_episode_length == 0, Episode<EnvironmentType>, Episode<EnvironmentType, max_episode_length>>;
  using KeyMaker = typename VALUE_FUNCTION_T::KeyMaker;

  // Generate an episode following policy pi
  EpisodeType episode;
  if constexpr (max_episode_length == 0) {
    // It is assumed that the terminal state will be generated in the normal
    // operation of the environments step funciton.
    episode = generate_episode(environment, policy);
  } else if constexpr (max_episode_length != 0) {
    // Either the step function will generate an episode termination or the
    // maximum episode length will be reached - and so the episode will stop
    // early.
    episode = generate_episode<max_episode_length>(environment, policy);
  }

  // Initialise the return to 0
  typename VALUE_FUNCTION_T::PrecisionType G = 0;
  // Loop over the transitions in the episode in reverse order
  for (auto it = episode.GetTransitions().rbegin(); it != episode.GetTransitions().rend(); ++it) {
    // Update the return
    G = EnvironmentType::RewardType::reward(*it) + valueFunction.discount_rate * G;

    // If the state does not appear in an earlier transition
    if (stop_condition.template operator()<EpisodeType>(episode.GetTransitions().begin(), (it + 1).base(), *it)) {
      const auto key = KeyMaker::make(it->state, it->action);
      // Add the return to the list of returns
      returns[key].push_back(G);
      // Update the value function to be the average of returns from that
      // state
      valueFunction[it->state].value =
          std::accumulate(returns[key].begin(), returns[key].end(), 0.0F) / returns[key].size();
    }
  }
}

/** @brief The value v_{pi}(s) is the average of all returns following the first
 * visit to s. */
template <std::size_t max_episode_length,
          policy::isFiniteStateValueFunction VALUE_FUNCTION_T,
          policy::isDistributionPolicy POLICY_T>
void first_visit_valueEstimate(VALUE_FUNCTION_T &valueFunction,
                               typename VALUE_FUNCTION_T::EnvironmentType &environment,
                               POLICY_T &policy,
                               std::size_t episodes) {

  // initialise the value function and returns list for every state
  auto returns = n_visit_returns_initialisation(valueFunction, environment);

  // Loop forever over episodes - Here we actually only loop for the requested
  // number of episodes
  for (std::size_t i = 0; i < episodes; ++i) {
    visit_valueEstimate_step<max_episode_length>(
        valueFunction, environment, policy, returns, FirstVisitStopCondition<VALUE_FUNCTION_T>());
  }
}

template <policy::isFiniteStateValueFunction VALUE_FUNCTION_T>
struct EveryVisitStopCondition : StopCondition<VALUE_FUNCTION_T> {
  // We always update the average at every visit.
  template <typename EPISODE_T>
  bool operator()(typename EPISODE_T::DataContainer::const_iterator start,
                  typename EPISODE_T::DataContainer::const_iterator end,
                  const typename VALUE_FUNCTION_T::TransitionType &transition) const {
    return true;
  }
};

/** @brief The value v_{pi}(s) is the average of all returns following all
 * visits to s. */
template <std::size_t max_episode_length,
          policy::isFiniteStateValueFunction VALUE_FUNCTION_T,
          policy::isDistributionPolicy POLICY_T>
void every_visit_valueEstimate(VALUE_FUNCTION_T &valueFunction,
                               typename VALUE_FUNCTION_T::EnvironmentType &environment,
                               POLICY_T &policy,
                               std::size_t episodes) {

  // initialise the value function and returns list for every state
  auto returns = n_visit_returns_initialisation(valueFunction, environment);

  // Loop forever over episodes - Here we actually only loop for the requested
  // number of episodes
  for (std::size_t i = 0; i < episodes; ++i) {
    visit_valueEstimate_step<max_episode_length>(
        valueFunction, environment, policy, returns, EveryVisitStopCondition<VALUE_FUNCTION_T>());
  }
}

} // namespace monte_carlo
