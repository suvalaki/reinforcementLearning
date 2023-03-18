#include "catch.hpp"
#include <cmath>
#include <iostream>

#include "markov_decision_process/coin_mdp.hpp"
#include "temporal_difference/value_iteration.hpp"
#include "temporal_difference/value_update/td0_updater.hpp"

using namespace temporal_difference;

TEST_CASE("temporal_difference::one_step_valueEstimate_episode") {

  // Because the coin MDP has a fixed reward then the updated reward will always be that fixed reward. 1.0

  auto data = CoinModelDataFixture{};
  auto &[s0, s1, a0, a1, transitionModel, environ, policySA, policyState, policyAction, _v0, valueFunction, _v2] = data;
  // auto returns = monte_carlo::n_visit_returns_initialisation(valueFunction, environ);
  auto updater = TD0Updater<std::decay_t<decltype(valueFunction)>>();
  updater.initialize(environ, valueFunction);
  one_step_valueEstimate_episode(
      valueFunction,
      environ,
      policySA,
      policySA, // This does nothing under this updater. Because this updater is on-policy
      updater,
      0.5F, // discount rate
      1000);

  // Value update should make the result non-zero. The values start at zero
  REQUIRE(valueFunction.valueAt(s0) != Approx(0.0));
  REQUIRE(valueFunction.valueAt(s1) != Approx(0.0));
}
