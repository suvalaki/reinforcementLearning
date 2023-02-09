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

template <std::size_t max_episode_length,
          policy::isFiniteStateValueFunction VALUE_FUNCTION_T,
          policy::isGreedyPolicy POLICY_T0,
          policy::isGreedyPolicy POLICY_T1,
          isStopCondition STOP_CONDITION_T,
          isEpisodeGenerator EPISODE_GENERATOR_T =
              EpisodeGenerator<max_episode_length, typename VALUE_FUNCTION_T::EnvironmentType, POLICY_T0>>
void visit_valueEstimate_step(
    VALUE_FUNCTION_T &valueFunction,
    typename VALUE_FUNCTION_T::EnvironmentType &environment,
    POLICY_T0 &policy,
    POLICY_T1 &target_policy,
    AverageReturnsMap<VALUE_FUNCTION_T> &returns,
    const STOP_CONDITION_T &stop_condition,
    const EPISODE_GENERATOR_T &episodeGenerator =
        EpisodeGenerator<max_episode_length, typename VALUE_FUNCTION_T::EnvironmentType, POLICY_T0>()) {

  using EnvironmentType = typename VALUE_FUNCTION_T::EnvironmentType;
  using KeyMaker = typename VALUE_FUNCTION_T::KeyMaker;

  auto episode = episodeGenerator(environment, policy);

  // Initialise the return to 0
  typename VALUE_FUNCTION_T::PrecisionType G = 0;
  // Loop over the transitions in the episode in reverse order
  for (auto it = episode.GetTransitions().rbegin(); it != episode.GetTransitions().rend(); ++it) {

    if (it->isDone())
      continue; // SKIP this is a terminal state. The agent and values have no meaning here.

    // Update the return
    G = EnvironmentType::RewardType::reward(*it) + valueFunction.discount_rate * G;

    // If the state does not appear in an earlier transition
    if (stop_condition.template operator()<typename EPISODE_GENERATOR_T::EpisodeType>(
            episode.GetTransitions().begin(), (it + 1).base(), *it)) {

      const auto key = KeyMaker::make(it->state, it->action);
      // Add the return to the list of returns
      // weight them by the potential off policy method. (5.3) and (5.4)
      auto importance_sampling_weight = &policy == &target_policy
                                            ? 1.0F
                                            : target_policy.getProbability(environment, it->state, key) /
                                                  policy.getProbability(environment, it->state, key);
      returns[key].push_back(importance_sampling_weight * G);
      // Update the value function to be the average of returns from that
      // state
      valueFunction[key].value = std::accumulate(returns[key].begin(), returns[key].end(), 0.0F) / returns[key].size();
      // Add the number of rewards being counted (for the average - step is available in our default Value
      // implementation)
      valueFunction[key].step = returns[key].size();
    }
  }
}

template <std::size_t max_episode_length,
          policy::isFiniteStateValueFunction VALUE_FUNCTION_T,
          policy::isGreedyPolicy POLICY_T0,
          policy::isGreedyPolicy POLICY_T1,
          isStopCondition STOP_CONDITION_T,
          isEpisodeGenerator EPISODE_GENERATOR_T =
              EpisodeGenerator<max_episode_length, typename VALUE_FUNCTION_T::EnvironmentType, POLICY_T0>>
void visit_valueEstimate(
    VALUE_FUNCTION_T &valueFunction,
    typename VALUE_FUNCTION_T::EnvironmentType &environment,
    POLICY_T0 &policy,
    POLICY_T1 &target_policy,
    std::size_t episodes,
    const STOP_CONDITION_T &stopCondition,
    const EPISODE_GENERATOR_T &episodeGenerator =
        EpisodeGenerator<max_episode_length, typename VALUE_FUNCTION_T::EnvironmentType, POLICY_T0>()) {

  // initialise the value function and returns list for every state
  auto returns = n_visit_returns_initialisation(valueFunction, environment);

  // Loop forever over episodes - Here we actually only loop for the requested
  // number of episodes
  for (std::size_t i = 0; i < episodes; ++i) {
    visit_valueEstimate_step<max_episode_length>(
        valueFunction, environment, policy, target_policy, returns, stopCondition, episodeGenerator);
  }
}

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

/** @brief The value v_{pi}(s) is the average of all returns following the first
 * visit to s. */
template <std::size_t max_episode_length,
          policy::isFiniteStateValueFunction VALUE_FUNCTION_T,
          policy::isGreedyPolicy POLICY_T0,
          policy::isGreedyPolicy POLICY_T1,
          isEpisodeGenerator EPISODE_GENERATOR_T =
              EpisodeGenerator<max_episode_length, typename VALUE_FUNCTION_T::EnvironmentType, POLICY_T0>>
void first_visit_valueEstimate(
    VALUE_FUNCTION_T &valueFunction,
    typename VALUE_FUNCTION_T::EnvironmentType &environment,
    POLICY_T0 &policy,
    POLICY_T1 &target_policy,
    std::size_t episodes,
    const EPISODE_GENERATOR_T &episodeGenerator =
        EpisodeGenerator<max_episode_length, typename VALUE_FUNCTION_T::EnvironmentType, POLICY_T0>()) {

  visit_valueEstimate<max_episode_length>(valueFunction,
                                          environment,
                                          policy,
                                          target_policy,
                                          episodes,
                                          FirstVisitStopCondition<VALUE_FUNCTION_T>(),
                                          episodeGenerator);
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
          policy::isGreedyPolicy POLICY_T0,
          policy::isGreedyPolicy POLICY_T1,
          isEpisodeGenerator EPISODE_GENERATOR_T =
              EpisodeGenerator<max_episode_length, typename VALUE_FUNCTION_T::EnvironmentType, POLICY_T0>>
void every_visit_valueEstimate(
    VALUE_FUNCTION_T &valueFunction,
    typename VALUE_FUNCTION_T::EnvironmentType &environment,
    POLICY_T0 &policy,
    POLICY_T1 &target_policy,
    std::size_t episodes,
    const EPISODE_GENERATOR_T &episodeGenerator =
        EpisodeGenerator<max_episode_length, typename VALUE_FUNCTION_T::EnvironmentType, POLICY_T0>()) {

  visit_valueEstimate<max_episode_length>(valueFunction,
                                          environment,
                                          policy,
                                          target_policy,
                                          episodes,
                                          EveryVisitStopCondition<VALUE_FUNCTION_T>(),
                                          episodeGenerator);
}

} // namespace monte_carlo
