#pragma once

#include <unordered_map>
#include <vector>

#include "monte_carlo/episode.hpp"
#include "monte_carlo/value_update/average_return.hpp"
#include "monte_carlo/value_update/value_update.hpp"
#include "policy/distribution_policy.hpp"
#include "policy/finite/value_policy.hpp"
#include "policy/objectives/finite_value_function.hpp"
#include "policy/value.hpp"

namespace monte_carlo {

// The SIMPLEST monte carlo control takes a Key over the state types. But this may not be best when
// we dont know the state dynamics.

/// Each occurance of a state within an episode is called a visit.

template <policy::objectives::isFiniteStateValueFunction VALUE_FUNCTION_T>
struct StopCondition {
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

template <
    std::size_t max_episode_length,
    policy::objectives::isFiniteStateValueFunction VALUE_FUNCTION_T,
    policy::isFinitePolicyValueFunctionMixin POLICY_T0,
    policy::isFinitePolicyValueFunctionMixin POLICY_T1,
    isStopCondition STOP_CONDITION_T,
    isValueUpdater VALUE_UPDATER_T = NiaveAverageReturnsUpdate<VALUE_FUNCTION_T>,
    isEpisodeGenerator EPISODE_GENERATOR_T =
        EpisodeGenerator<max_episode_length, typename VALUE_FUNCTION_T::EnvironmentType, POLICY_T0>>
requires(std::is_same_v<typename VALUE_FUNCTION_T::KeyMaker, typename POLICY_T0::KeyMaker> &&std::is_same_v<
         typename POLICY_T0::KeyMaker,
         typename POLICY_T1::KeyMaker>) //
    void visit_valueEstimate_step(
        VALUE_FUNCTION_T &valueFunction,
        typename VALUE_FUNCTION_T::EnvironmentType &environment,
        POLICY_T0 &policy,
        POLICY_T1 &target_policy,
        VALUE_UPDATER_T &valueUpdater,
        const STOP_CONDITION_T &stop_condition,
        const EPISODE_GENERATOR_T &episodeGenerator =
            EpisodeGenerator<max_episode_length, typename VALUE_FUNCTION_T::EnvironmentType, POLICY_T0>()) {

  SETUP_TYPES_W_VALUE_FUNCTION(VALUE_FUNCTION_T);

  auto episode = episodeGenerator(environment, policy);

  // initialize the return to 0
  typename VALUE_FUNCTION_T::PrecisionType G = 0;
  // Loop over the transitions in the episode in reverse order
  for (auto it = episode.GetTransitions().rbegin(); it != episode.GetTransitions().rend(); ++it) {

    if (it->isDone())
      continue; // SKIP this is a terminal state. The agent and values have no meaning here.

    // Update the return
    G = EnvironmentType::RewardType::reward(*it) + valueFunction.discount_rate * G;

    // If the state does not appear in an earlier transition
    if (stop_condition.template operator()<typename EPISODE_GENERATOR_T::EpisodeType>(
            environment, episode.GetTransitions().begin(), (it + 1).base(), *it)) {
      valueUpdater.update(valueFunction, policy, target_policy, environment, it->state, it->action, G);
    }
  }
}

template <
    std::size_t max_episode_length,
    policy::objectives::isFiniteStateValueFunction VALUE_FUNCTION_T,
    policy::isFinitePolicyValueFunctionMixin POLICY_T0,
    policy::isFinitePolicyValueFunctionMixin POLICY_T1,
    isStopCondition STOP_CONDITION_T,
    typename VALUE_UPDATER_T = NiaveAverageReturnsUpdate<VALUE_FUNCTION_T>,
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

  // initialize the value function and returns list for every state
  // auto returns = n_visit_returns_initialisation(valueFunction, environment);
  auto valueUpdater = VALUE_UPDATER_T();

  // Loop forever over episodes - Here we actually only loop for the requested
  // number of episodes
  for (std::size_t i = 0; i < episodes; ++i) {
    visit_valueEstimate_step<max_episode_length>(
        valueFunction, environment, policy, target_policy, valueUpdater, stopCondition, episodeGenerator);
  }
}

template <policy::objectives::isFiniteStateValueFunction VALUE_FUNCTION_T>
struct FirstVisitStopCondition : StopCondition<VALUE_FUNCTION_T> {

  SETUP_TYPES_W_VALUE_FUNCTION(VALUE_FUNCTION_T);

  // Only update the average for the first visit in the episode
  template <typename EPISODE_T>
  bool operator()(
      typename EPISODE_T::EnvironmentType &environment,
      typename EPISODE_T::DataContainer::const_iterator start,
      typename EPISODE_T::DataContainer::const_iterator end,
      const typename VALUE_FUNCTION_T::EnvironmentType::TransitionType &transition) const {
    // if any of the earlier transitions have the key then they make
    // a better first visit.
    return not std::any_of(start, end, [&environment, &transition](const auto &t) {
      return KeyMaker::make(environment, t.state, t.action) ==
             KeyMaker::make(environment, transition.state, transition.action);
    });
  }
};

/** @brief The value v_{pi}(s) is the average of all returns following the first
 * visit to s. */
template <
    std::size_t max_episode_length,
    policy::objectives::isFiniteStateValueFunction VALUE_FUNCTION_T,
    policy::isFinitePolicyValueFunctionMixin POLICY_T0,
    policy::isFinitePolicyValueFunctionMixin POLICY_T1,
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

  visit_valueEstimate<max_episode_length>(
      valueFunction,
      environment,
      policy,
      target_policy,
      episodes,
      FirstVisitStopCondition<VALUE_FUNCTION_T>(),
      episodeGenerator);
}

template <policy::objectives::isFiniteStateValueFunction VALUE_FUNCTION_T>
struct EveryVisitStopCondition : StopCondition<VALUE_FUNCTION_T> {
  // We always update the average at every visit.
  template <typename EPISODE_T>
  bool operator()(
      typename EPISODE_T::EnvironmentType &environment,
      typename EPISODE_T::DataContainer::const_iterator start,
      typename EPISODE_T::DataContainer::const_iterator end,
      const typename VALUE_FUNCTION_T::TransitionType &transition) const {
    return true;
  }
};

/** @brief The value v_{pi}(s) is the average of all returns following all
 * visits to s. */
template <
    std::size_t max_episode_length,
    policy::objectives::isFiniteStateValueFunction VALUE_FUNCTION_T,
    policy::isFinitePolicyValueFunctionMixin POLICY_T0,
    policy::isFinitePolicyValueFunctionMixin POLICY_T1,
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

  visit_valueEstimate<max_episode_length>(
      valueFunction,
      environment,
      policy,
      target_policy,
      episodes,
      EveryVisitStopCondition<VALUE_FUNCTION_T>(),
      episodeGenerator);
}

} // namespace monte_carlo
