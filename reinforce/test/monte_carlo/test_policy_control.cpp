#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <iostream>

#include <reinforce/monte_carlo/policy_control.hpp>
#include <reinforce/monte_carlo/value.hpp>
#include <reinforce/policy/epsilon_greedy_policy.hpp>
#include <reinforce/policy/finite/epsilon_greedy_policy.hpp>
#include <reinforce/policy/finite/greedy_policy.hpp>
#include <reinforce/policy/finite/random_policy.hpp>
#include <reinforce/policy/greedy_policy.hpp>
#include <reinforce/policy/objectives/value_function_keymaker.hpp>

#include "markov_decision_process/coin_mdp.hpp"

using namespace Catch;

using KeyMaker = policy::objectives::StateActionKeymaker<CoinEnviron>;
using GreedyCoinPolicy = policy::FiniteGreedyPolicy<CoinFiniteStateActionValueFunction>;
using RandomCoinPolicy = policy::FiniteRandomPolicy<CoinEnviron>;
using EpsGreedyCoinPolicy = policy::FiniteEpsilonGreedyPolicy<RandomCoinPolicy, GreedyCoinPolicy>;

TEST_CASE("monte_carlo::monte_carlo_on_policy_first_visit_control_with_exploring_starts") {

  SECTION("On Policy Control - first visit") {

    auto data = CoinModelDataFixture{};
    auto &[s0, s1, a0, a1, transitionModel, environ, policy, policyState, policyAction, valueFunction, _v1, _v2] = data;

    auto greedyPolicy = GreedyCoinPolicy();

    // Prior to running the policy the estimates are all zero.
    monte_carlo::monte_carlo_on_policy_first_visit_control_with_exploring_starts<10>(greedyPolicy, environ, 15);

    // No garuntee that the stat action pair is selected, but at least one should be.
    auto values = std::vector<float>{
        greedyPolicy[{s0, a0}].value,
        greedyPolicy[{s1, a0}].value,
        greedyPolicy[{s0, a1}].value,
        greedyPolicy[{s1, a1}].value};
    CHECK(std::any_of(values.begin(), values.end(), [](auto v) { return v != Approx(0.0); }));

    auto randomPolicy = RandomCoinPolicy();
    auto epsGreedyPolicy = EpsGreedyCoinPolicy{randomPolicy, {}, 0.2F};
    // std::cout << greedyPolicy.getProbability(environ, s0, a0) << "\n";
    epsGreedyPolicy.getProbability(environ, s0, a0);

    for (int i = 0; i < 10; ++i) {
      // std::cout << epsGreedyPolicy(environ, s0) << "\n";
      // std::cout << static_cast<typename EpsGreedyCoinPolicy::ExploitType &>(epsGreedyPolicy)(environ, s0) << "\n";
    }

    // epsGreedyPolicy.getProbability(environ, s0, a0);

    // Prior to running the policy the estimates are all zero.
    monte_carlo::monte_carlo_on_policy_first_visit_control_with_exploring_starts<10>(epsGreedyPolicy, environ, 50);

    auto valuesEps = std::vector<float>{
        epsGreedyPolicy[{s0, a0}].value,
        epsGreedyPolicy[{s1, a0}].value,
        epsGreedyPolicy[{s0, a1}].value,
        epsGreedyPolicy[{s1, a1}].value};
    CHECK(std::any_of(valuesEps.begin(), valuesEps.end(), [](auto v) { return v != Approx(0.0); }));
  }

  SECTION("OFF Policy Control with importance sampling") {

    auto data = CoinModelDataFixture{};
    auto &[s0, s1, a0, a1, transitionModel, environ, policy, policyState, policyAction, valueFunction, _v1, _v2] = data;

    {
      auto greedyPolicy = GreedyCoinPolicy();
      auto randomPolicy = RandomCoinPolicy();
      auto epsGreedyPolicy = EpsGreedyCoinPolicy(randomPolicy, {});

      // We are attempting to learn the greedy policy using the epsilon greedy policy.

      // Start by training the epsilon greedy policy and then use that to train the greedy policy via off policy
      // control. monte_carlo::monte_carlo_on_policy_first_visit_control_with_exploring_starts<10>(epsGreedyPolicy,
      // environ, 50);

      // Prior to running the policy the estimates are all zero.
      monte_carlo::monte_carlo_off_policy_importance_sampling_first_visit_control_with_exploring_starts<10>(
          epsGreedyPolicy, // control policy
          greedyPolicy,    // target policy
          environ,
          5);

      auto values = std::vector<float>{
          greedyPolicy[{s0, a0}].value,
          greedyPolicy[{s1, a0}].value,
          greedyPolicy[{s0, a1}].value,
          greedyPolicy[{s1, a1}].value};
      CHECK(std::any_of(values.begin(), values.end(), [](auto v) { return v != Approx(0.0); }));
    }

    {
      auto greedyPolicy = GreedyCoinPolicy();
      auto randomPolicy = RandomCoinPolicy();
      auto epsGreedyPolicy = EpsGreedyCoinPolicy(randomPolicy, {});

      // Prior to running the policy the estimates are all zero.
      monte_carlo::monte_carlo_off_policy_importance_sampling_every_visit_control_with_exploring_starts<10>(
          epsGreedyPolicy, greedyPolicy, environ, 5);

      auto values = std::vector<float>{
          greedyPolicy[{s0, a0}].value,
          greedyPolicy[{s1, a0}].value,
          greedyPolicy[{s0, a1}].value,
          greedyPolicy[{s1, a1}].value};
      CHECK(std::any_of(values.begin(), values.end(), [](auto v) { return v != Approx(0.0); }));
    }
  }
}

TEST_CASE("monte_carlo::monte_carlo_on_policy_every_visit_control_with_exploring_starts") {

  auto data = CoinModelDataFixture{};
  auto &[s0, s1, a0, a1, transitionModel, environ, policy, policyState, policyAction, valueFunction, _v1, _v2] = data;

  auto greedyPolicy = GreedyCoinPolicy();

  // // Prior to running the policy the estimates are all zero.
  // monte_carlo::monte_carlo_on_policy_every_visit_control_with_exploring_starts<10>(greedyPolicy, environ, 5);

  // CHECK((greedyPolicy[{s0, a0}].value) != Approx(0.0));
  // CHECK((greedyPolicy[{s1, a0}].value) != Approx(0.0));
  // CHECK((greedyPolicy[{s0, a1}].value) != Approx(0.0));
  // CHECK((greedyPolicy[{s1, a1}].value) != Approx(0.0));

  // using EpsGreedyCoinPolicy = policy::EpsilonGreedyPolicy<CoinEnviron, GreedyCoinPolicy>;
  // auto epsGreedyPolicy = GreedyCoinPolicy();

  // // Prior to running the policy the estimates are all zero.
  // monte_carlo::monte_carlo_on_policy_every_visit_control_with_exploring_starts<10>(epsGreedyPolicy, environ, 5);

  // CHECK((epsGreedyPolicy[{s0, a0}].value) != Approx(0.0));
  // CHECK((epsGreedyPolicy[{s1, a0}].value) != Approx(0.0));
  // CHECK((epsGreedyPolicy[{s0, a1}].value) != Approx(0.0));
  // CHECK((epsGreedyPolicy[{s1, a1}].value) != Approx(0.0));
}