#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <iostream>

#include <reinforce/environment.hpp>
#include <reinforce/monte_carlo/value.hpp>
#include <reinforce/policy/objectives/value_function_keymaker.hpp>
#include <reinforce/policy/value.hpp>

#include "markov_decision_process/coin_mdp.hpp"

using namespace environment;

TEST_CASE("Test Value Function", "[policy][value]") {

  auto data = CoinModelDataFixture{};
  auto &[s0, s1, a0, a1, transitionModel, environ, policy, policyState, policyAction, _v0, valueFunction, _v2] = data;

  auto stateValueFunction = CoinFiniteStateValueFunction{};
  auto stateActionValueFunction = CoinFiniteStateActionValueFunction{};
  auto finiteStateActionValueFunction = CoinFiniteStateActionValueFunction{};
  auto finiteStateValueFunction = CoinFiniteStateValueFunction{};
  auto finiteActionValueFunction = CoinFiniteActionValueFunction{};

  finiteStateValueFunction.initialize(environ);
  finiteStateActionValueFunction.initialize(environ);
  finiteActionValueFunction.initialize(environ);

  finiteStateActionValueFunction.prettyPrint();
  finiteStateValueFunction.prettyPrint();
  finiteActionValueFunction.prettyPrint();
}