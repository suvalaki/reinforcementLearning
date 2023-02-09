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

  SETUP_TYPES_FROM_ENVIRON(SINGLE_ARG(ENVIRON_T));
  using BaseType = Policy<EnvironmentType>;

  // Run the policy over the current state of the environment
  virtual ActionSpace operator()(const StateType &s) = 0;
  virtual void update(const TransitionType &s) = 0;
};

template <typename T>
concept PolicyType = std::is_base_of_v<Policy<typename T::EnvironmentType>, T>;

template <environment::EnvironmentType ENVIRON_T, isStateActionKeymaker KEYMAPPER_T = DefaultActionKeymaker<ENVIRON_T>>
struct FiniteStatePolicy {

  SETUP_TYPES_FROM_ENVIRON(SINGLE_ARG(ENVIRON_T));
  using BaseType = Policy<EnvironmentType>;

  // Run the policy over the current state of the environment
  virtual ActionSpace operator()(const StateType &s) = 0;
  virtual void update(const TransitionType &s) = 0;
};

} // namespace policy
