#pragma once
#include <utility>

#include "environment.hpp"

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

} // namespace monte_carlo