#pragma once

#include <functional>
#include <type_traits>
#include <vector>

#include "environment.hpp"
#include "policy/policy.hpp"
#include "transition.hpp"

namespace monte_carlo {

template <typename T, std::size_t N = 0>
class Episode;

template <environment::EnvironmentType ENVIRONMENT_T>
class Episode<ENVIRONMENT_T> {
public:
  using EnvironmentType = ENVIRONMENT_T;
  using DataContainer = std::vector<typename ENVIRONMENT_T::TransitionType>;

  void AddTransition(const typename ENVIRONMENT_T::TransitionType &transition) { transitions_.push_back(transition); }

  const DataContainer &GetTransitions() const { return transitions_; }

private:
  DataContainer transitions_;
};

// Template Specialistaion - for compile time episode.
template <environment::EnvironmentType ENVIRONMENT_T, std::size_t episode_size>
class Episode<ENVIRONMENT_T, episode_size> {
public:
  using EnvironmentType = ENVIRONMENT_T;
  using DataContainer = transition::TransitionSequence<episode_size, typename ENVIRONMENT_T::ActionSpace>;

  void AddTransition(const typename ENVIRONMENT_T::TransitionType &transition) {
    if (currIdx_ >= episode_size) {
      throw std::runtime_error("Episode is full");
    }
    transitions_[currIdx_++] = transition;
  }

  const DataContainer &GetTransitions() const { return transitions_; }

private:
  DataContainer transitions_;
  std::size_t currIdx_ = 0;
};

template <typename T>
concept isEpisode = requires(T t) {
  typename T::EnvironmentType;
  {t.AddTransition()};
  { t.GetTransitions() } -> std::convertible_to<typename T::DataContainer>;
};

template <environment::EnvironmentType ENVIRONMENT_T, policy::implementsPolicy POLICY_T>
Episode<ENVIRONMENT_T> generate_episode_base(
    ENVIRONMENT_T &environment,
    POLICY_T &policy,
    const std::function<bool(const typename ENVIRONMENT_T::TransitionType &)> &stop_condition) {
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

template <environment::EnvironmentType ENVIRONMENT_T, policy::implementsPolicy POLICY_T>
Episode<ENVIRONMENT_T> generate_episode(ENVIRONMENT_T &environment, POLICY_T &policy, const bool reset = true) {

  auto episode = Episode<ENVIRONMENT_T>{};
  if (reset)
    environment.reset();
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
};

template <std::size_t T>
concept NonZero = (T > 0);

template <std::size_t episode_max_length, environment::EnvironmentType ENVIRONMENT_T, policy::implementsPolicy POLICY_T>
requires NonZero<episode_max_length> Episode<ENVIRONMENT_T, episode_max_length>
generate_episode(ENVIRONMENT_T &environment, POLICY_T &policy, const bool reset = true) {

  auto episode = Episode<ENVIRONMENT_T, episode_max_length>{};
  if (reset)
    environment.reset();
  auto state = environment.getState();

  std::size_t insertion = 0;
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

template <
    std::size_t max_episode_length_n,
    environment::EnvironmentType ENVIRONMENT_T,
    policy::implementsPolicy POLICY_T>
struct EpisodeGenerator {

  static std::size_t constexpr max_episode_length = max_episode_length_n;

  using EnvironmentType = ENVIRONMENT_T;
  using PolicyType = POLICY_T;
  using EpisodeType = std::
      conditional_t<max_episode_length == 0, Episode<EnvironmentType>, Episode<EnvironmentType, max_episode_length>>;

  virtual EpisodeType operator()(ENVIRONMENT_T &environment, POLICY_T &policy) const {
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
    return episode;
  }
};

template <typename T>
concept isEpisodeGenerator = std::
    is_base_of<EpisodeGenerator<T::max_episode_length, typename T::EnvironmentType, typename T::PolicyType>, T>::value;

} // namespace monte_carlo
