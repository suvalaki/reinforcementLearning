#pragma once
#include <tuple>

#include "policy/policy.hpp"

namespace policy {

/// @brief A base template interface for use in defining different types of combinations of policies.
template <implementsPolicy... POLICY_T> struct PolicyCombination {

  using PolicyTypes = std::tuple<POLICY_T...>;
  SETUP_TYPES_FROM_NESTED_ENVIRON(SINGLE_ARG(std::tuple_element_t<0, PolicyTypes>::EnvironmentType));

  using PolicyRefTuple = std::tuple<std::reference_wrapper<POLICY_T>...>;
  PolicyRefTuple policies;

  PolicyCombination(POLICY_T &...policies) : policies(policies...) {}

  // How the combination is applied is an implementation detail for the specific combination.
};

} // namespace policy
