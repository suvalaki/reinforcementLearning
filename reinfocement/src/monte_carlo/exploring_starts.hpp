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

template <environment::FiniteEnvironmentType ENVIRONMENT_T>
typename ENVIRONMENT_T::TransitionType exploring_start_init(ENVIRONMENT_T &e) {
  auto [state, action] = exploring_start_sample(e);
  e.reset();
  e.state = state;
  return e.step(action);
  // Now follow policy \pi
}

template <environment::EnvironmentType ENVIRONMENT_T, policy::isFinitePolicyValueFunctionMixin POLICY_T>
Episode<ENVIRONMENT_T> generate_episode_with_exploring_starts(ENVIRONMENT_T &environment, POLICY_T &policy) {

  std::size_t insertion = 0;
  auto episode = Episode<ENVIRONMENT_T>{};
  {
    auto transition = exploring_start_init(environment);
    episode.AddTransition(transition);
    insertion++;
    if (transition.isDone()) {
      return episode;
    }
  }

  auto state = environment.getState();
  while (true) {
    auto action = policy(environment, state);
    auto transition = environment.step(action);
    episode.AddTransition(transition);
    if (transition.isDone()) {
      break;
    }
    state = transition.nextState;
    environment.update(transition);
  }
  return episode;
}

template <std::size_t episode_max_length,
          environment::EnvironmentType ENVIRONMENT_T,
          policy::isFinitePolicyValueFunctionMixin POLICY_T>
requires NonZero<episode_max_length> Episode<ENVIRONMENT_T, episode_max_length>
generate_episode_with_exploring_starts(ENVIRONMENT_T &environment, POLICY_T &policy) {

  std::size_t insertion = 0;
  auto episode = Episode<ENVIRONMENT_T, episode_max_length>{};
  {
    auto transition = exploring_start_init(environment);
    episode.AddTransition(transition);
    insertion++;
    if (transition.isDone() || insertion >= episode_max_length) {
      return episode;
    }
  }

  auto state = environment.getState();
  while (true) {
    auto action = policy(environment, state);
    auto transition = environment.step(action);
    episode.AddTransition(transition);
    insertion++;
    if (transition.isDone() || insertion >= episode_max_length) {
      break;
    }
    state = transition.nextState;
    environment.update(transition);
  }
  return episode;
}

template <std::size_t max_episode_length_n,
          environment::EnvironmentType ENVIRONMENT_T,
          policy::isFinitePolicyValueFunctionMixin POLICY_T>
struct ExploringStartsEpisodeGenerator : EpisodeGenerator<max_episode_length_n, ENVIRONMENT_T, POLICY_T> {

  using EpisodeType = EpisodeGenerator<max_episode_length_n, ENVIRONMENT_T, POLICY_T>::EpisodeType;

  EpisodeType operator()(ENVIRONMENT_T &environment, POLICY_T &policy) const override {
    if constexpr (max_episode_length_n == 0) {
      // It is assumed that the terminal state will be generated in the normal
      // operation of the environments step funciton.
      return generate_episode_with_exploring_starts(environment, policy);
    } else if constexpr (max_episode_length_n != 0) {
      // Either the step function will generate an episode termination or the
      // maximum episode length will be reached - and so the episode will stop
      // early.
      return generate_episode_with_exploring_starts<max_episode_length_n>(environment, policy);
    }
  }
};

} // namespace monte_carlo