#include "catch.hpp"
#include <cmath>
#include <iostream>

#include "environment_fixtures.hpp"
#include "markov_decision_process/coin_mdp.hpp"
#include "monte_carlo/value.hpp"

using namespace monte_carlo;

TEST_CASE("monte_carlo::NiaveAverageReturnsUpdate") {
  auto data = CoinModelDataFixture{};
  auto &[s0, s1, a0, a1, transitionModel, environ, policy, policyState, policyAction, _v0, valueFunction, _v2] = data;
  auto updater = monte_carlo::NiaveAverageReturnsUpdate<std::decay_t<decltype(valueFunction)>>();

  // Average returns should still work when empty
  updater.updateReturns(valueFunction, policy, policy, environ, s0, a0, 1);
  auto key = decltype(updater)::KeyMaker::make(environ, s0, a0);
  auto averageReturn0 = updater.getAverageReturn(key);
  REQUIRE(averageReturn0 == 1);
  updater.updateValue(valueFunction, policy, policy, environ, s0, a0);
  REQUIRE(valueFunction[key].value == 1);

  updater.updateReturns(valueFunction, policy, policy, environ, s0, a0, 2);
  auto averageReturn1 = updater.getAverageReturn(key);
  REQUIRE(averageReturn1 == 1.5);
  updater.updateValue(valueFunction, policy, policy, environ, s0, a0);
  REQUIRE(valueFunction[key].value == 1.5);
}

TEST_CASE("monte_carlo::NiaveAverageReturnsIncrementalUpdate") {
  auto data = CoinModelDataFixture{};
  auto &[s0, s1, a0, a1, transitionModel, environ, policy, policyState, policyAction, _v0, valueFunction, _v2] = data;
  auto updater = monte_carlo::NiaveAverageReturnsIncrementalUpdate<std::decay_t<decltype(valueFunction)>>();

  // Average returns should still work when empty
  updater.updateReturns(valueFunction, policy, policy, environ, s0, a0, 1);
  auto key = decltype(updater)::KeyMaker::make(environ, s0, a0);
  REQUIRE(updater.returns[key].averageReturn == 1);
  updater.updateValue(valueFunction, policy, policy, environ, s0, a0);
  REQUIRE(valueFunction[key].value == 1);

  updater.updateReturns(valueFunction, policy, policy, environ, s0, a0, 2);
  REQUIRE(updater.returns[key].averageReturn == 1.5);
  updater.updateValue(valueFunction, policy, policy, environ, s0, a0);
  REQUIRE(valueFunction[key].value == 1.5);
}
