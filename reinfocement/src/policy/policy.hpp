#pragma once
#include "action.hpp"
#include "environment.hpp"
#include <random>
#include <xtensor/xfixed.hpp>
#include <xtensor/xrandom.hpp>

#include "action.hpp"
#include "spec.hpp"
#include "state_action_keymaker.hpp"
#include "state_action_value.hpp"
#include "step_size.hpp"

namespace policy {

using spec::CompositeArraySpecType;
using spec::isBoundedArraySpec;
using spec::isCategoricalArraySpec;

template <environment::EnvironmentType ENVIRON_T> struct Policy {
  using EnvironmentType = ENVIRON_T;
  using PrecisionType = typename EnvironmentType::PrecisionType;
  using StateType = typename EnvironmentType::StateType;
  using ActionSpace = typename EnvironmentType::ActionSpace;
  using TransitionType = typename EnvironmentType::TransitionType;

  // Run the policy over the current state of the environment
  virtual ActionSpace operator()(const StateType &s) = 0;
  virtual void update(const TransitionType &s) = 0;
};

template <typename T>
concept PolicyType = std::is_base_of_v<Policy<typename T::EnvironmentType>, T>;

} // namespace policy
