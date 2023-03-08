#pragma once
#include <random>
#include <xtensor/xfixed.hpp>
#include <xtensor/xrandom.hpp>

#include "action.hpp"
#include "environment.hpp"
#include "spec.hpp"
#include "state_action_value.hpp"

namespace policy {

// A policy might also be a value function or a distribution over actions from a given state. A policy could also just
// be some completely balck box that takes a state and returns an action. This is the base class for all policies. As a
// simplification we assume all policies are both a value function and a distribution at the same time. However this
// might not necesarily always be the case - for example a random policy has no need for values.

using spec::CompositeArraySpecType;
using spec::isBoundedArraySpec;
using spec::isCategoricalArraySpec;

template <environment::EnvironmentType ENVIRON_T>
struct Policy {

  SETUP_TYPES_FROM_ENVIRON(SINGLE_ARG(ENVIRON_T));
  using BaseType = Policy<EnvironmentType>;

  // Run the policy over the current state of the environment
  virtual ActionSpace operator()(const EnvironmentType &e, const StateType &s) const = 0;
  virtual void update(const EnvironmentType &e, const TransitionType &s) = 0;
};

template <typename T>
concept implementsPolicy = std::is_base_of_v<Policy<typename T::EnvironmentType>, T>;

/// @brief Default methods to mix into a policy to setup an interface for a distribution over actions from a given
/// state.
template <environment::EnvironmentType ENVIRON_T>
struct PolicyDistributionMixin {

  SETUP_TYPES_FROM_ENVIRON(SINGLE_ARG(ENVIRON_T));

  // Within a given state query the probability of the policy taking a particular action.
  virtual PrecisionType getProbability(const EnvironmentType &e, const StateType &s, const ActionSpace &a) const = 0;
  virtual PrecisionType getLogProbability(const EnvironmentType &e, const StateType &s, const ActionSpace &a) const = 0;
  virtual PrecisionType getKernel(const EnvironmentType &e, const StateType &s, const ActionSpace &a) const = 0;
  virtual PrecisionType getNormalisationConstant(const EnvironmentType &e, const StateType &s) const = 0;

  // Sample an action from the policy distribution.
  virtual ActionSpace sampleAction(const EnvironmentType &e, const StateType &s) const = 0;
  virtual ActionSpace getArgmaxAction(const EnvironmentType &e, const StateType &s) const = 0;

  /// @brief Assuming this is the target policy what is the importance ratio against the observed policy
  template <typename DISTRIBUTION_POLICY_T>
  PrecisionType importanceSamplingRatio(
      const EnvironmentType &e, const StateType &s, const ActionSpace &a, const DISTRIBUTION_POLICY_T &other) const;
};

template <environment::EnvironmentType E>
template <typename DISTRIBUTION_POLICY_T>
auto PolicyDistributionMixin<E>::importanceSamplingRatio(
    const EnvironmentType &e, const StateType &s, const ActionSpace &a, const DISTRIBUTION_POLICY_T &other) const
    -> PrecisionType {
  if (this == &other)
    return 1.0;
  return getProbability(e, s, a) / other.getProbability(e, s, a);
}

template <typename T>
concept implementsPolicyDistributionMixin = std::is_base_of_v<PolicyDistributionMixin<typename T::EnvironmentType>, T>;

} // namespace policy
