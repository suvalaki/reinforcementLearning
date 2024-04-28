#pragma once

#include "reinforce/environment.hpp"

/// @brief A set oof dummy templates for use in creating template concepts.
namespace environment::dummy {

using DummyState = state::State<float>;
using DummyAction = action::Action<DummyState>;
using DummyStep = step::Step<DummyAction>;
using DummyReward = reward::Reward<DummyAction>;
using DummyReturn = returns::Return<DummyReward>;
using DummyEnvironment = Environment<DummyStep, DummyReward, DummyReturn>;
using DummyFiniteEnvironment = FiniteEnvironment<DummyStep, DummyReward, DummyReturn>;

} // namespace environment::dummy
