#pragma once

#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "reinforce/environment.hpp"
#include "reinforce/policy/combination_policy.hpp"
#include "reinforce/policy/finite/policy.hpp"
#include "reinforce/policy/objectives/finite_value.hpp"
#include "reinforce/policy/objectives/finite_value_function.hpp"
#include "reinforce/policy/objectives/value_function_keymaker.hpp"
#include "reinforce/policy/value.hpp"

#define FVT FinitePolicyValueFunctionMixin<VALUE_FUNCTION_T>

namespace policy {

// The distinguishing feature of a finite value policy is the ability to hold mappins from
// all of the keys in the environment to values.

template <objectives::isFiniteValueFunction VALUE_FUNCTION_T>
// requires std::is_same_v<typename KEYMAPPER_T::EnvironmentType, typename VALUE_T::EnvironmentType>
struct FinitePolicyValueFunctionMixin : virtual PolicyDistributionMixin<typename VALUE_FUNCTION_T::EnvironmentType>,
                                        PolicyValueFunctionMixin<VALUE_FUNCTION_T> {

  using BaseType = VALUE_FUNCTION_T;
  SETUP_TYPES_W_VALUE_FUNCTION(VALUE_FUNCTION_T);
  using ValueFunctionBaseType = typename ValueFunctionType::ValueFunctionBaseType;
  using StepSizeTaker = typename ValueFunctionType::StepSizeTaker;

  FinitePolicyValueFunctionMixin(auto &&...args);
  FinitePolicyValueFunctionMixin(const FinitePolicyValueFunctionMixin &p);

  using ValueFunctionType::initialize;
  ActionSpace getArgmaxAction(const EnvironmentType &e, const StateType &s) const override;
};

template <objectives::isFiniteValueFunction VALUE_FUNCTION_T>
FVT::FinitePolicyValueFunctionMixin(auto &&...args) : PolicyValueFunctionMixin<VALUE_FUNCTION_T>(args...) {}

template <objectives::isFiniteValueFunction VALUE_FUNCTION_T>
FVT::FinitePolicyValueFunctionMixin(const FinitePolicyValueFunctionMixin &p)
    : PolicyValueFunctionMixin<VALUE_FUNCTION_T>(p) {}

template <objectives::isFiniteValueFunction VALUE_FUNCTION_T>
auto FVT::getArgmaxAction(const EnvironmentType &e, const StateType &s) const -> ActionSpace {
  return KeyMaker::get_action_from_key(e, this->getArgmaxKey(e, s));
}

template <typename T>
concept isFinitePolicyValueFunctionMixin =
    std::is_base_of_v<FinitePolicyValueFunctionMixin<typename T::ValueFunctionType>, T>;

template <typename T>
concept implementsFiniteValuePolicy = isFinitePolicyValueFunctionMixin<T> && isFinitePolicy<T>;

} // namespace policy

#undef FVT