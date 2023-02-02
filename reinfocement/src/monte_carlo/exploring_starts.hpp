#pragma once
#include <utility>

#include "environment.hpp"

#include "monte_carlo/episode.hpp"
#include "monte_carlo/value.hpp"

namespace monte_carlo {

// In monte carlo ES (exploring starts) we start from a random state and then follow the policy
// in order to estimate qvals (state-action values).

/// @brief Generate an episode starting from a random state-action pair (admissible in thee environment)
template <environment::FiniteEnvironmentType ENVIRONMENT_T>
std::pair<typename ENVIRONMENT_T::StateType, typename ENVIRONMENT_T::ActionSpace>
exploring_start_sample(const ENVIRONMENT_T &e) {

  const auto state = e.randomState();
  const auto action = e.randomAction(state);
  return std::make_pair(state, action);
}

template <environment::FiniteEnvironmentType ENVIRONMENT_T> void exploring_start_init(const ENVIRONMENT_T &e) {
  auto [state, action] = exploring_start_sample(e);
  e.reset();
  e.state = state;
  e.step(action);
  // Now follow policy \pi
}

template <environment::EnvironmentType ENVIRONMENT_T, policy::PolicyType POLICY_T>
Episode<ENVIRONMENT_T> generate_episode_with_exploring_starts(ENVIRONMENT_T &environment, POLICY_T &policy) {
  exploring_start_init(environment);
  auto episode = generate_episode(environment, policy, false);
  return episode;
}

template <std::size_t episode_max_length, environment::EnvironmentType ENVIRONMENT_T, policy::PolicyType POLICY_T>
Episode<ENVIRONMENT_T> generate_episode_with_exploring_starts(ENVIRONMENT_T &environment, POLICY_T &policy) {
  exploring_start_init(environment);
  auto episode = generate_episode<episode_max_length>(environment, policy, false);
  return episode;
}

template <std::size_t max_episode_length, environment::EnvironmentType ENVIRONMENT_T, policy::PolicyType POLICY_T>
struct ExploringStartsEpisodeGenerator : EpisodeGenerator<max_episode_length, ENVIRONMENT_T, POLICY_T> {

  using EpisodeType = EpisodeGenerator<max_episode_length, ENVIRONMENT_T, POLICY_T>::EpisodeType;

  EpisodeType operator()(ENVIRONMENT_T &environment, POLICY_T &policy) const override {
    EpisodeType episode;
    if constexpr (max_episode_length == 0) {
      // It is assumed that the terminal state will be generated in the normal
      // operation of the environments step funciton.
      episode = generate_episode_with_exploring_starts(environment, policy);
    } else if constexpr (max_episode_length != 0) {
      // Either the step function will generate an episode termination or the
      // maximum episode length will be reached - and so the episode will stop
      // early.
      episode = generate_episode_with_exploring_starts<max_episode_length>(environment, policy);
    }
    return episode;
  }
};

} // namespace monte_carlo