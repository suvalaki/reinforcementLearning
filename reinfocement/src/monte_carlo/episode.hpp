#pragma once

#include <vector>

#include "environment.hpp"
#include "transition.hpp"

namespace monte_carlo {

template <environment::EnvironmentType ENVIRONMENT_T> class Episode {
public:
  using EnvironmentType = ENVIRONMENT_T;

  void AddTransition(const typename ENVIRONMENT_T::TransitionType &transition) {
    transitions_.push_back(transition);
  }

  const std::vector<typename ENVIRONMENT_T::TransitionType> &
  GetTransitions() const {
    return transitions_;
  }

private:
  std::vector<typename ENVIRONMENT_T::TransitionType> transitions_;
};

template <typename T>
concept isEpisode = requires(T t) {
  typename T::EnvironmentType;
  {t.AddTransition()};
  {
    t.GetTransitions()
    } -> std::convertible_to<
        std::vector<typename T::EnvironmentType::TransitionType>>;
};

template <environment::EnvironmentType ENVIRONMENT_T,
          policy::PolicyType POLICY_T>
Episode<ENVIRONMENT_T> generate_episode_base(
    ENVIRONMENT_T &environment, POLICY_T &policy,
    const std::function<bool(const typename ENVIRONMENT_T::TransitionType &)>
        &stop_condition) {
  auto episode = Episode<ENVIRONMENT_T>{};
  auto state = environment.reset();
  while (true) {
    auto action = policy(state);
    auto transition = environment.step(action);
    episode.AddTransition(transition);
    if (stop_condition(transition)) {
      break;
    }
    state = transition.next_state;
    environment.update(transition);
  }
  return episode;
}

template <environment::EnvironmentType ENVIRONMENT_T,
          policy::PolicyType POLICY_T>
Episode<ENVIRONMENT_T> generate_episode(ENVIRONMENT_T &environment,
                                        POLICY_T &policy) {
  auto episode = Episode<ENVIRONMENT_T>{};
  auto state = environment.reset();
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

template <environment::EnvironmentType ENVIRONMENT_T,
          policy::PolicyType POLICY_T>
Episode<ENVIRONMENT_T> generate_episode(ENVIRONMENT_T &environment,
                                        POLICY_T &policy,
                                        std::size_t episode_max_length) {
  auto episode = Episode<ENVIRONMENT_T>{};
  auto state = environment.reset();
  while (true) {
    auto action = policy(environment, state);
    auto transition = environment.step(action);
    episode.AddTransition(transition);
    if (transition.isDone() ||
        episode.GetTransitions().size() >= episode_max_length) {
      break;
    }
    state = transition.nextState;
    environment.update(transition);
  }
  return episode;
}

} // namespace monte_carlo
