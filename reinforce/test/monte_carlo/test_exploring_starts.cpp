#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <iostream>

#include <reinforce/monte_carlo/exploring_starts.hpp>
#include <reinforce/monte_carlo/value.hpp>

#include "markov_decision_process/coin_mdp.hpp"

TEST_CASE("monte_carlo::exploring_start_initialisation") {

  auto data = CoinModelDataFixture{};
  auto &[s0, s1, a0, a1, transitionModel, environ, policy, policyState, policyAction, valueFunction, _v1, _v2] = data;
  auto [state, action] = monte_carlo::exploring_start_sample(environ);
  REQUIRE((state == s0 or state == s1));
  REQUIRE((action == a0 or action == a1));
};
