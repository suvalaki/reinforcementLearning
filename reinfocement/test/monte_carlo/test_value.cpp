#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>
#include <iostream>

#include "markov_decision_process/coin_mdp.hpp"
#include "monte_carlo/value.hpp"

TEST_CASE("monte_carlo::FirstVisitStopCondition") {}

TEST_CASE("monte_carlo::EveryVisitStopCondition") {}

TEST_CASE("monte_carlo::visit_valueEstimate_step") {
  auto data = CoinModelDataFixture{};
  auto &[s0, s1, a0, a1, transitionModel, environ, policy, policyState, policyAction, _v0, valueFunction, _v2] = data;
  // auto returns = monte_carlo::n_visit_returns_initialisation(valueFunction, environ);
  auto updater = monte_carlo::NiaveAverageReturnsUpdate<std::decay_t<decltype(valueFunction)>>();
  monte_carlo::visit_valueEstimate_step<10>(
      valueFunction,
      environ,
      policyState,
      policyState, // Without importance sampling
      updater,
      monte_carlo::FirstVisitStopCondition<std::decay_t<decltype(valueFunction)>>());
  REQUIRE(updater.returns.size() == 2);
  REQUIRE(((updater.returns[s0].size() == 1) or (updater.returns[s1].size() == 1)));
}

TEST_CASE("monte_carlo::first_visit_valueEstimate") {
  auto data = CoinModelDataFixture{};
  auto &[s0, s1, a0, a1, transitionModel, environ, policy, policyState, policyAction, _v0, valueFunction, _v2] = data;
  monte_carlo::first_visit_valueEstimate<10>(valueFunction, environ, policyState, policyState, 10);
  // Updates were made - because the rewards are always positive the values
  // will also be positive and non zero.
  // Because of randomness it can be the case that one of the states is never
  // visited
  REQUIRE(((valueFunction[s0] > 0.0F) or (valueFunction[s1] > 0.0F)));
  REQUIRE_FALSE(std::isnan(valueFunction[s0].value));
  REQUIRE_FALSE(std::isnan(valueFunction[s1].value));
}

TEST_CASE("monte_carlo::every_visit_valueEstimate") {
  auto data = CoinModelDataFixture{};
  auto &[s0, s1, a0, a1, transitionModel, environ, policy, policyState, policyAction, _v0, valueFunction, _v2] = data;
  monte_carlo::every_visit_valueEstimate<10>(valueFunction, environ, policyState, policyState, 10);
  // Updates were made - because the rewards are always positive the values
  // will also be positive and non zero.
  REQUIRE(((valueFunction[s0] > 0.0F) or (valueFunction[s1] > 0.0F)));
  REQUIRE_FALSE(std::isnan(valueFunction[s0].value));
  REQUIRE_FALSE(std::isnan(valueFunction[s1].value));
}