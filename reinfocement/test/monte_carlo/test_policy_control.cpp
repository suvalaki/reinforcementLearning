#include "catch.hpp"
#include <cmath>
#include <iostream>

#include "markov_decision_process/coin_mdp.hpp"
#include "monte_carlo/policy_control.hpp"
#include "monte_carlo/value.hpp"
#include "policy/epsilon_greedy_policy.hpp"
#include "policy/greedy_policy.hpp"

TEST_CASE("monte_carlo::monte_carlo_control_with_exploring_starts") {

  auto data = CoinModelDataFixture{};
  auto &[s0, s1, a0, a1, transitionModel, environ, policy, valueFunction] = data;

  using GreedyCoinPolicy = policy::GreedyPolicy<CoinEnviron>;
  auto greedyPolicy = GreedyCoinPolicy();

  // Prior to running the policy the estimates are all zero.
  monte_carlo::monte_carlo_control_with_exploring_starts<10>(greedyPolicy, environ, 150);

  CHECK((greedyPolicy[{s0, a0}].value) != Approx(0.0));
  CHECK((greedyPolicy[{s1, a0}].value) != Approx(0.0));
  CHECK((greedyPolicy[{s0, a1}].value) != Approx(0.0));
  CHECK((greedyPolicy[{s1, a1}].value) != Approx(0.0));

  using EpsGreedyCoinPolicy = policy::EpsilonGreedyPolicy<CoinEnviron, GreedyCoinPolicy>;
  auto epsGreedyPolicy = GreedyCoinPolicy();

  // Prior to running the policy the estimates are all zero.
  monte_carlo::monte_carlo_control_with_exploring_starts<10>(epsGreedyPolicy, environ, 150);

  CHECK((epsGreedyPolicy[{s0, a0}].value) != Approx(0.0));
  CHECK((epsGreedyPolicy[{s1, a0}].value) != Approx(0.0));
  CHECK((epsGreedyPolicy[{s0, a1}].value) != Approx(0.0));
  CHECK((epsGreedyPolicy[{s1, a1}].value) != Approx(0.0));
}