#include "catch.hpp"
#include <cmath>
#include <iostream>

#include "markov_decision_process/coin_mdp.hpp"
#include "temporal_difference/value_iteration.hpp"
#include "temporal_difference/value_update/td0_updater.hpp"

using namespace temporal_difference;

TEST_CASE("temporal_difference::TD0Updater") {

  // Because the coin MDP has a fixed reward then the updated reward will always be that fixed reward. 1.0

  auto data = CoinModelDataFixture{};
  auto &[s0, s1, a0, a1, transitionModel, environ, policySA, policyState, policyAction, _v0, valueFunction, _v2] = data;
  // auto returns = monte_carlo::n_visit_returns_initialisation(valueFunction, environ);
  auto updater = TD0Updater<std::decay_t<decltype(valueFunction)>>();
  updater.initialize(environ, valueFunction);
  updater.update(valueFunction,
                 policySA,
                 policySA, // This does nothing under this updater. Because this updater is on-policy
                 environ,
                 a0,
                 0.5F // discount rate
  );

  // Values all start at zero. After the reset we start at state s0. we can only have updated s0 so far...
  // We know the reward is 1.0 no matter the action or next state. So the value update is defined by:
  // V(S) = V(S) + alpha * (R + gamma * V(S') - V(S))
  //      = 0.0 + alpha * (1.0 + (0.5) 0.0 - 0.0) = 1.0
  // Sometimes the reward can be -1.0 as well if we land on s1 instead of s0 (its a random env).
  CHECK(((valueFunction.valueAt(s0) == Approx(1.0)) or (valueFunction.valueAt(s0) == Approx(-1.0))));
  CHECK(valueFunction.valueAt(s1) == Approx(0.0));
}
