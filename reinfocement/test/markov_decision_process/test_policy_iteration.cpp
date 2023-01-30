#include "catch.hpp"
#include <iostream>

#include "coin_mdp.hpp"
#include "markov_decision_process/finite_transition_model.hpp"
#include "markov_decision_process/policy_iteration.hpp"
#include "policy/distribution_policy.hpp"
#include "policy/random_policy.hpp"

using namespace environment;
using namespace markov_decision_process;

TEST_CASE("Coin MPD can undergo policy evaluation") {

  auto data = CoinModelDataFixture{};
  auto &[s0, s1, a0, a1, transitionModel, environ, policy, valueFunction] =
      data;

  policy.printQTable(environ);

  SECTION("Several iterations of value iteration steps succesfully update the "
          "value") {
    auto initialValue = valueFunction.valueAt(s0);
    for (int i = 0; i < 100; ++i) {
      auto val = policy_evaluation_step(valueFunction, environ, policy, s0);
      // std::cout << val << "\n";
      valueFunction.at(s0) = val;
      // valueFunction.policy_improvement_step(environ, policy, s0);
    }

    // Validate that the value iteration has indeed updated the value
    CHECK_FALSE(initialValue == valueFunction.at(s0).value);
  }

  SECTION("Complete pass over all states works") {
    // because we need to compare the values we must at least initialise them.
    valueFunction.initialize(environ);
    auto initialValues = valueFunction;
    // valueFunction.policy_evaluation(environ, policy, 1e-3F);
    policy_evaluation(valueFunction, environ, policy, 1e-3F);
    for (auto &[state, value] : valueFunction) {
      CHECK(value != initialValues.at(state));
    }
  }
}

TEST_CASE("Coin MPD can undergo policy improvement") {

  auto data = CoinModelDataFixture{};
  auto &[s0, s1, a0, a1, transitionModel, environ, policy, valueFunction] =
      data;

  SECTION("Policy improvement step updates a policy") {
    // force the q_table to have non-optimal values
    policy.q_table.at(CoinDistributionPolicy::KeyMaker::make(s0, a0)) = 1.0F;
    policy.q_table.at(CoinDistributionPolicy::KeyMaker::make(s0, a1)) = 0.0F;
    policy.q_table.at(CoinDistributionPolicy::KeyMaker::make(s1, a0)) = 0.0F;
    policy.q_table.at(CoinDistributionPolicy::KeyMaker::make(s1, a1)) = 0.0F;
    policy_evaluation(valueFunction, environ, policy, 1e-3F);
    auto updated = policy_improvement_step(valueFunction, environ, policy, s0);
    CHECK_FALSE(updated); // the policy updated
    std::cout << policy.getProbability(
                     environ, s0,
                     CoinDistributionPolicy::KeyMaker::make(s0, a0))
              << "\n";
    std::cout << policy.getProbability(
                     environ, s0,
                     CoinDistributionPolicy::KeyMaker::make(s0, a1))
              << "\n";

    policy.printQTable(environ);
  }

  SECTION("Complete Policy improvement") {
    // force the q_table to have non-optimal values
    policy.q_table.at(CoinDistributionPolicy::KeyMaker::make(s0, a0)) = 1.0F;
    policy.q_table.at(CoinDistributionPolicy::KeyMaker::make(s0, a1)) = 0.0F;
    policy.q_table.at(CoinDistributionPolicy::KeyMaker::make(s1, a0)) = 0.0F;
    policy.q_table.at(CoinDistributionPolicy::KeyMaker::make(s1, a1)) = 0.0F;
    policy_evaluation(valueFunction, environ, policy, 1e-3F);
    policy_improvement(valueFunction, environ, policy);
    auto p = policy.getProbability(
        environ, s0, CoinDistributionPolicy::KeyMaker::make(s0, a0));
    CHECK(p != Approx(1.0F));
  }
}

TEST_CASE("Coin MPD can undergo policy iteration") {

  auto data = CoinModelDataFixture{};
  auto &[s0, s1, a0, a1, transitionModel, environ, policy, valueFunction] =
      data;

  // force the q_table to have non-optimal values
  policy.q_table.at(CoinDistributionPolicy::KeyMaker::make(s0, a0)) = 1.0F;
  policy.q_table.at(CoinDistributionPolicy::KeyMaker::make(s0, a1)) = 1.0F;
  policy.q_table.at(CoinDistributionPolicy::KeyMaker::make(s1, a0)) = 0.0F;
  policy.q_table.at(CoinDistributionPolicy::KeyMaker::make(s1, a1)) = 0.0F;
  policy_iteration(valueFunction, environ, policy, 1e-3F);
  auto p = policy.getProbability(
      environ, s0, CoinDistributionPolicy::KeyMaker::make(s0, a0));
  CHECK(p != Approx(1.0F));

  policy.printQTable(environ);
}