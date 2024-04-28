#pragma once
#include <cmath>
#include <exception>
#include <limits>
#include <random>
#include <xtensor/xfixed.hpp>
#include <xtensor/xrandom.hpp>

#include "reinforce/action.hpp"
#include "reinforce/environment.hpp"
#include "reinforce/policy/greedy_policy.hpp"
#include "reinforce/policy/random_policy.hpp"
#include "reinforce/policy/state_action_value.hpp"
#include "reinforce/spec.hpp"

namespace policy {

template <environment::EnvironmentType E>
struct DistributionPolicy : virtual PolicyDistributionMixin<E>, Policy<E> {

  using BaseType = Policy<E>;
  SETUP_TYPES_FROM_NESTED_ENVIRON(SINGLE_ARG(BaseType::EnvironmentType));

  // For a distribution policy we just sample from the policy distribution as the action
  ActionSpace operator()(const EnvironmentType &e, const StateType &s) const override;

  // The other overrides will need to be implemented
};

template <environment::EnvironmentType E>
typename DistributionPolicy<E>::ActionSpace
DistributionPolicy<E>::operator()(const EnvironmentType &e, const StateType &s) const {
  return this->sampleAction(e, s);
}

template <typename T>
concept isDistributionPolicy = std::is_base_of_v<DistributionPolicy<typename T::EnvironmentType>, T>;

} // namespace policy
