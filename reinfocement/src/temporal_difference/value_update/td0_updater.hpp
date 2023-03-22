#pragma once
#include <utility>

#include "policy/distribution_policy.hpp"
#include "policy/finite/value_policy.hpp"
#include "policy/objectives/finite_value_function.hpp"
#include "policy/value.hpp"

#include "temporal_difference/value_update/q_learning.hpp"
#include "temporal_difference/value_update/value_update.hpp"

namespace temporal_difference {

// This is basically Q learning when valueFunction == policy
template <policy::objectives::isFiniteStateValueFunction V>
requires policy::objectives::isStateKeymaker<typename V::KeyMaker>
using TD0Updater = TemporalDifferenceValueUpdater<V, QLearningStepMixin, DefaultValueUpdater>;

} // namespace temporal_difference
