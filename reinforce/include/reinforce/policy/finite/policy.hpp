#pragma once

#include "reinforce/policy/objectives/value_function_keymaker.hpp"
#include "reinforce/policy/policy.hpp"

namespace policy {

template <environment::FiniteEnvironmentType E>
using FinitePolicy = Policy<E>;
template <environment::FiniteEnvironmentType E>
using FinitePolicyDistributionMixin = PolicyDistributionMixin<E>;

template <typename T>
concept isFinitePolicy = std::is_base_of_v<FinitePolicy<typename T::EnvironmentType>, T>;

template <isFinitePolicy P>
struct FinitePolicyIncrementalMixin {};

} // namespace policy
