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

  // Lets note that the value function starts empty and only adds values to itself when the state index is first 
  // requested of it (and then the value would default to zero). This is a computational efficiency decision.

  // Validate that running an episode of length 10 moves the values appropriately
  // The first reachable method means that state value is (from the end of the episode) the return 
  // of all future rewards from that point (where the terminal state doesnt count for updating; so at least 
  // 1 discount timestep).  
  // Because of the nature of this transition matrix the states can always return to themselves so it is possible
  // to only ever update a single one, or to update both . 
  monte_carlo::first_visit_valueEstimate<10>(valueFunction, environ, policyState, policyState, 10);

  REQUIRE(valueFunction.size() == 2); // 2 possible states
  // For simplicty we (for now) can just check that at least one state moved away from its initial value. 
  // TODO: add the actual mathermatical cases here?
  REQUIRE((
    ((valueFunction[s0] != 0.0F) and (valueFunction[s0] != -1.0F) and (valueFunction[s0] != 1.0F) )  
    or 
    ((valueFunction[s1] != 0.0F) and (valueFunction[s1] != -1.0F) and (valueFunction[s1] != 1.0F) )  
  ));
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