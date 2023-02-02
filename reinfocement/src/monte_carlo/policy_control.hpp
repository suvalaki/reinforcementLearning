#pragma once

#include "monte_carlo/exploring_starts.hpp"
#include "monte_carlo/value.hpp"

namespace monte_carlo {

// Monte carlo constrol is about maximusing average(Return(s,a)) over all states and actions under pi.
template <std::size_t episode_size, policy::isGreedyPolicy POLICY_T>
void monte_carlo_control_with_exploring_starts(POLICY_T &policy,
                                               typename POLICY_T::EnvironmentType &environment,
                                               std::size_t episodes) {

  using PolicyType = POLICY_T;
  using EnvironmentType = typename POLICY_T::EnvironmentType;
  auto episodeGenerator = ExploringStartsEpisodeGenerator<episode_size, EnvironmentType, PolicyType>{};

  // NOTE - The greedy policy is ALSO a value function. and so can be provided to this function. The choice
  // of greedy outcome will AUTOMATICALLY update.

  // Initialise valueFunction with Q(key) forall keys (estimate q(s,a) or q(key)))
  // Initialise the value function (and policy) to the initial value

  // Over the episode update the value estimates
  // Update the policy to be greedy with respect to the value function - automatic for the policy as it
  // also holds the value function.
  first_visit_valueEstimate<episode_size>(
      // dynamic_cast<typename PolicyType::ValueFunctionType &>(policy), environment, policy, episodes,
      // episodeGenerator);
      policy,
      environment,
      policy,
      episodes,
      episodeGenerator);
}

} // namespace monte_carlo
