#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <iostream>

#include <reinforce/temporal_difference/n_step.hpp>

#include "markov_decision_process/coin_mdp.hpp"

using namespace temporal_difference;

TEST_CASE("temporal_difference::n_stepReturn") {

  auto data = CoinModelDataFixture{};
  auto &[s0, s1, a0, a1, transitionModel, environ, policySA, policyState, policyAction, valueFunction, _v1, _v2] = data;

  // Create the transitions where reward is always 1 . discount rate doesnt exist
  auto t0 = typename CoinEnviron::TransitionType{.state = s0, .action = a0, .nextState = s1};

  auto ret0 = nStepReturn(environ, 1.0F, t0, t0, t0);
  CHECK(ret0 == 1.0F + 1.0F + 1.0F);

  auto ret1 = nStepReturn(environ, 0.5F, t0, t0, t0);
  CHECK(ret1 == 1.0F + 0.5F + 0.25F);
}
