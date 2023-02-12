#include "catch.hpp"
#include <cmath>
#include <iostream>

#include "markov_decision_process/coin_mdp.hpp"
#include "monte_carlo/policy_control.hpp"
#include "monte_carlo/value.hpp"
#include "policy/epsilon_greedy_policy.hpp"
#include "policy/greedy_policy.hpp"

TEST_CASE("monte_carlo::monte_carlo_on_policy_first_visit_control_with_exploring_starts") {

  SECTION("On Policy Control - first visit") {

    auto data = CoinModelDataFixture{};
    auto &[s0, s1, a0, a1, transitionModel, environ, policy, policyState, policyAction, valueFunction, _v1, _v2] = data;

    using GreedyCoinPolicy = policy::GreedyPolicy<CoinEnviron>;
    auto greedyPolicy = GreedyCoinPolicy();

    // Prior to running the policy the estimates are all zero.
    monte_carlo::monte_carlo_on_policy_first_visit_control_with_exploring_starts<10>(greedyPolicy, environ, 15);

    // No garuntee that the stat action pair is selected, but at least one should be.
    auto values = std::vector<float>{greedyPolicy[{s0, a0}].value,
                                     greedyPolicy[{s1, a0}].value,
                                     greedyPolicy[{s0, a1}].value,
                                     greedyPolicy[{s1, a1}].value};
    CHECK(std::any_of(values.begin(), values.end(), [](auto v) { return v != Approx(0.0); }));

    using EpsGreedyCoinPolicy = policy::EpsilonGreedyPolicy<CoinEnviron, GreedyCoinPolicy>;
    auto epsGreedyPolicy = GreedyCoinPolicy();

    // Prior to running the policy the estimates are all zero.
    monte_carlo::monte_carlo_on_policy_first_visit_control_with_exploring_starts<10>(epsGreedyPolicy, environ, 50);

    auto valuesEps = std::vector<float>{epsGreedyPolicy[{s0, a0}].value,
                                        epsGreedyPolicy[{s1, a0}].value,
                                        epsGreedyPolicy[{s0, a1}].value,
                                        epsGreedyPolicy[{s1, a1}].value};
    CHECK(std::any_of(valuesEps.begin(), valuesEps.end(), [](auto v) { return v != Approx(0.0); }));
  }

  SECTION("OFF Policy Control with importance sampling") {

    auto data = CoinModelDataFixture{};
    auto &[s0, s1, a0, a1, transitionModel, environ, policy, policyState, policyAction, valueFunction, _v1, _v2] = data;

    {
      using GreedyCoinPolicy = policy::GreedyPolicy<CoinEnviron>;
      using EpsGreedyCoinPolicy = policy::EpsilonGreedyPolicy<CoinEnviron, GreedyCoinPolicy>;
      auto greedyPolicy = GreedyCoinPolicy();
      auto epsGreedyPolicy = EpsGreedyCoinPolicy();

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

      auto values = std::vector<float>{greedyPolicy[{s0, a0}].value,
                                       greedyPolicy[{s1, a0}].value,
                                       greedyPolicy[{s0, a1}].value,
                                       greedyPolicy[{s1, a1}].value};
      CHECK(std::any_of(values.begin(), values.end(), [](auto v) { return v != Approx(0.0); }));
      greedyPolicy.printQTable();
    }

    {
      using GreedyCoinPolicy = policy::GreedyPolicy<CoinEnviron>;
      using EpsGreedyCoinPolicy = policy::EpsilonGreedyPolicy<CoinEnviron, GreedyCoinPolicy>;
      auto greedyPolicy = GreedyCoinPolicy();
      auto epsGreedyPolicy = EpsGreedyCoinPolicy();

      // Prior to running the policy the estimates are all zero.
      monte_carlo::monte_carlo_off_policy_importance_sampling_every_visit_control_with_exploring_starts<10>(
          epsGreedyPolicy, greedyPolicy, environ, 5);

      auto values = std::vector<float>{greedyPolicy[{s0, a0}].value,
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

  using GreedyCoinPolicy = policy::GreedyPolicy<CoinEnviron>;
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