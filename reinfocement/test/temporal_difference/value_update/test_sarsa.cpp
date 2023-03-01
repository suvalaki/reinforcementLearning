#include "catch.hpp"
#include <cmath>
#include <iostream>

#include "markov_decision_process/coin_mdp.hpp"
#include "temporal_difference/value_iteration.hpp"
#include "temporal_difference/value_update/sarsa.hpp"

using namespace temporal_difference;

TEST_CASE("temporal_difference::SARSAUpdater") {

  // Because the coin MDP has a fixed reward then the updated reward will always be that fixed reward. 1.0

  auto data = CoinModelDataFixture{};
  auto &[s0, s1, a0, a1, transitionModel, environ, policySA, policyState, policyAction, valueFunction, _v1, _v2] = data;
  // auto returns = monte_carlo::n_visit_returns_initialisation(valueFunction, environ);
  auto updater = SARSAUpdater<std::decay_t<decltype(valueFunction)>>();
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
  CHECK(((valueFunction.valueAt(decltype(updater)::KeyMaker::make(environ, s0, a0)) != Approx(0.0)) or
         (valueFunction.valueAt(decltype(updater)::KeyMaker::make(environ, s0, a1)) != Approx(0.0))));
}
