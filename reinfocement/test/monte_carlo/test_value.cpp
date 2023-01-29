#include "catch.hpp"
#include <cmath>
#include <iostream>

#include "markov_decision_process/coin_mdp.hpp"
#include "monte_carlo/value.hpp"

TEST_CASE("monte_carlo::n_visit_retuns_initialisation") {
  auto data = CoinModelDataFixture{};
  auto &[s0, s1, a0, a1, transitionModel, environ, policy, valueFunction] =
      data;
  auto returns =
      monte_carlo::n_visit_returns_initialisation(valueFunction, environ);
  REQUIRE(returns.size() == 2);
  REQUIRE(returns[s0].size() == 0);
  REQUIRE(returns[s1].size() == 0);
}

TEST_CASE("monte_carlo::FirstVisitStopCondition") {}

TEST_CASE("monte_carlo::EveryVisitStopCondition") {}

TEST_CASE("monte_carlo::visit_valueEstimate_step") {
  auto data = CoinModelDataFixture{};
  auto &[s0, s1, a0, a1, transitionModel, environ, policy, valueFunction] =
      data;
  auto returns =
      monte_carlo::n_visit_returns_initialisation(valueFunction, environ);
  monte_carlo::visit_valueEstimate_step<10>(
      valueFunction, environ, policy, returns,
      monte_carlo::FirstVisitStopCondition<CoinEnviron>());
  REQUIRE(returns.size() == 2);
  REQUIRE(returns[s0].size() == 1);
  REQUIRE(returns[s1].size() == 1);
}

TEST_CASE("monte_carlo::first_visit_valueEstimate") {
  auto data = CoinModelDataFixture{};
  auto &[s0, s1, a0, a1, transitionModel, environ, policy, valueFunction] =
      data;
  monte_carlo::first_visit_valueEstimate<10>(valueFunction, environ, policy,
                                             10);
  // Updates were made - because the rewards are always positive the values
  // will also be positive and non zero.
  REQUIRE(valueFunction.valueEstimates[s0] > 0.0F);
  REQUIRE(valueFunction.valueEstimates[s1] > 0.0F);
  REQUIRE_FALSE(std::isnan(valueFunction.valueEstimates[s0]));
  REQUIRE_FALSE(std::isnan(valueFunction.valueEstimates[s1]));
}

TEST_CASE("monte_carlo::every_visit_valueEstimate") {
  auto data = CoinModelDataFixture{};
  auto &[s0, s1, a0, a1, transitionModel, environ, policy, valueFunction] =
      data;
  monte_carlo::every_visit_valueEstimate<10>(valueFunction, environ, policy,
                                             10);
  // Updates were made - because the rewards are always positive the values
  // will also be positive and non zero.
  REQUIRE(valueFunction.valueEstimates[s0] > 0.0F);
  REQUIRE(valueFunction.valueEstimates[s1] > 0.0F);
  REQUIRE_FALSE(std::isnan(valueFunction.valueEstimates[s0]));
  REQUIRE_FALSE(std::isnan(valueFunction.valueEstimates[s1]));
}