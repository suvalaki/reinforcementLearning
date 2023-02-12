#include "catch.hpp"
#include <cmath>
#include <iostream>

#include "markov_decision_process/coin_mdp.hpp"
#include "monte_carlo/value.hpp"

TEST_CASE("monte_carlo::n_visit_retuns_initialisation") {

  auto data = CoinModelDataFixture{};
  auto &[s0, s1, a0, a1, transitionModel, environ, policy, policyState, policyAction, _v0, valueFunction, _v2] = data;
  auto returns = monte_carlo::n_visit_returns_initialisation(valueFunction, environ);
  REQUIRE(returns.size() == 2);
  REQUIRE(returns[s0].size() == 0);
  REQUIRE(returns[s1].size() == 0);

  SECTION("Check isStateActionValueFunction overload") {
    auto data = CoinModelDataFixture{};
    auto &[s0, s1, a0, a1, transitionModel, environ, policy, policyState, policyAction, valueFunction, _v1, _v2] = data;
    auto vFuncActoinStates = CoinFiniteStateActionValueFunction();
    auto returns = monte_carlo::n_visit_returns_initialisation(vFuncActoinStates, environ);
    REQUIRE(returns.size() == 4); // One for each state action pair
    REQUIRE((returns[{s0, a0}].size() == 0));
    REQUIRE((returns[{s0, a1}].size() == 0));
    REQUIRE((returns[{s1, a0}].size() == 0));
    REQUIRE((returns[{s1, a1}].size() == 0));
  }
  SECTION("Check isStateValueFunction overload") {
    auto data = CoinModelDataFixture{};
    auto &[s0, s1, a0, a1, transitionModel, environ, policy, policyState, policyAction, _v0, valueFunction, _v2] = data;
    auto vFuncStates = CoinFiniteStateValueFunction();
    auto returns = monte_carlo::n_visit_returns_initialisation(vFuncStates, environ);
    REQUIRE((returns.size() == 2)); // one for each state
    REQUIRE((returns[s0].size() == 0));
    REQUIRE((returns[s1].size() == 0));
  }
  SECTION("Check isActionValueFunction overload") {
    auto data = CoinModelDataFixture{};
    auto &[s0, s1, a0, a1, transitionModel, environ, policy, policyState, policyAction, _v0, _v1, valueFunction] = data;
    auto vFuncActions = CoinFiniteActionValueFunction();
    auto returns = monte_carlo::n_visit_returns_initialisation(vFuncActions, environ);
    REQUIRE((returns.size() == 2)); // one for each action
    REQUIRE((returns[a0].size() == 0));
    REQUIRE((returns[a1].size() == 0));
  }
  // TODO - add tests for the other overloads
  SECTION("Check isNotKnownValueFunction overload") {}
}

TEST_CASE("monte_carlo::FirstVisitStopCondition") {}

TEST_CASE("monte_carlo::EveryVisitStopCondition") {}

TEST_CASE("monte_carlo::visit_valueEstimate_step") {
  auto data = CoinModelDataFixture{};
  auto &[s0, s1, a0, a1, transitionModel, environ, policy, policyState, policyAction, _v0, valueFunction, _v2] = data;
  auto returns = monte_carlo::n_visit_returns_initialisation(valueFunction, environ);
  auto returns_policy = monte_carlo::n_visit_returns_initialisation(policyState, environ); // need initd valuePolicy
  monte_carlo::visit_valueEstimate_step<10>(
      valueFunction,
      environ,
      policyState,
      policyState, // Without importance sampling
      returns,
      monte_carlo::FirstVisitStopCondition<std::decay_t<decltype(valueFunction)>>());
  REQUIRE(returns.size() == 2);
  REQUIRE(((returns[s0].size() == 1) or (returns[s1].size() == 1)));
}

TEST_CASE("monte_carlo::first_visit_valueEstimate") {
  auto data = CoinModelDataFixture{};
  auto &[s0, s1, a0, a1, transitionModel, environ, policy, policyState, policyAction, _v0, valueFunction, _v2] = data;
  monte_carlo::n_visit_returns_initialisation(policyState, environ); // need initd valuePolicy
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
  monte_carlo::n_visit_returns_initialisation(policyState, environ); // need initd valuePolicy
  monte_carlo::every_visit_valueEstimate<10>(valueFunction, environ, policyState, policyState, 10);
  // Updates were made - because the rewards are always positive the values
  // will also be positive and non zero.
  REQUIRE(((valueFunction[s0] > 0.0F) or (valueFunction[s1] > 0.0F)));
  REQUIRE_FALSE(std::isnan(valueFunction[s0].value));
  REQUIRE_FALSE(std::isnan(valueFunction[s1].value));
}