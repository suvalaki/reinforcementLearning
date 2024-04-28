#pragma once

#include <type_traits>

#include "reinforce/environment.hpp"
#include "reinforce/policy/epsilon_greedy_policy.hpp"
#include "reinforce/policy/finite/policy.hpp"
#include "reinforce/policy/finite/value_policy.hpp"
#include "reinforce/policy/policy.hpp"

namespace policy {

template <
    implementsPolicy EXPLORE_POLICY,
    implementsFiniteValuePolicy EXPLOIT_POLICY,
    class E = xt::random::default_engine_type>
struct FiniteEpsilonGreedyPolicy : EpsilonSoftPolicy<EXPLORE_POLICY, EXPLOIT_POLICY, E> {

  using BaseType = EXPLOIT_POLICY;
  using BaseEpsilonSoftType = EpsilonSoftPolicy<EXPLORE_POLICY, EXPLOIT_POLICY, E>;
  SETUP_TYPES_FROM_NESTED_ENVIRON(SINGLE_ARG(EXPLOIT_POLICY::EnvironmentType));
  using ExploreType = typename BaseEpsilonSoftType::ExploreType;
  using ExploitType = typename BaseEpsilonSoftType::ExploitType;
  using EngineType = typename BaseEpsilonSoftType::EngineType;

  FiniteEpsilonGreedyPolicy(
      const ExploreType &explorePolicy,
      const ExploitType &exploitPolicy,
      PrecisionType epsilon = 0.1,
      E &engine = xt::random::get_default_random_engine())
      : BaseEpsilonSoftType(explorePolicy, exploitPolicy, epsilon, engine) {}
};

template <typename T>
concept isFiniteEpsilonSoftPolicy = std::is_base_of_v<
    FiniteEpsilonGreedyPolicy<typename T::ExploreType, typename T::ExploitType, typename T::EngineType>,
    T>;

} // namespace policy
