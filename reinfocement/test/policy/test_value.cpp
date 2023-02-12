#include "catch.hpp"
#include <cmath>
#include <iostream>

#include "environment.hpp"
#include "markov_decision_process/coin_mdp.hpp"
#include "monte_carlo/value.hpp"
#include "policy/state_action_keymaker.hpp"
#include "policy/value.hpp"

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

  std::cout << "here\n";
  finiteStateActionValueFunction.prettyPrint();
  finiteStateValueFunction.prettyPrint();
  finiteActionValueFunction.prettyPrint();
}