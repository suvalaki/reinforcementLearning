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
  auto &[s0, s1, a0, a1, transitionModel, environ, policy, policyState, policyAction, _v0, valueFunction, _v1] = data;

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
  auto &[s0, s1, a0, a1, transitionModel, environ, policy, policyState, policyAction, _v0, valueFunction, _v1] = data;

  SECTION("Policy improvement step updates a policy") {
    // force the q_table to have non-optimal values
    policy.at(CoinDistributionPolicy::KeyMaker::make(environ, s0, a0)).value = 1.0F;
    policy.at(CoinDistributionPolicy::KeyMaker::make(environ, s0, a1)).value = 0.0F;
    policy.at(CoinDistributionPolicy::KeyMaker::make(environ, s1, a0)).value = 0.0F;
    policy.at(CoinDistributionPolicy::KeyMaker::make(environ, s1, a1)).value = 0.0F;
    policy_evaluation(valueFunction, environ, policy, 1e-3F);
    auto updated = policy_improvement_step(valueFunction, environ, policy, s0);
    CHECK_FALSE(updated); // the policy updated
    std::cout << policy.getProbability(environ, s0, a0) << "\n";
    std::cout << policy.getProbability(environ, s0, a1) << "\n";
  }

  SECTION("Complete Policy improvement") {
    // force the q_table to have non-optimal values
    policy.at(CoinDistributionPolicy::KeyMaker::make(environ, s0, a0)).value = 1.0F;
    policy.at(CoinDistributionPolicy::KeyMaker::make(environ, s0, a1)).value = 0.0F;
    policy.at(CoinDistributionPolicy::KeyMaker::make(environ, s1, a0)).value = 0.0F;
    policy.at(CoinDistributionPolicy::KeyMaker::make(environ, s1, a1)).value = 0.0F;
    policy_evaluation(valueFunction, environ, policy, 1e-3F);
    policy_improvement(valueFunction, environ, policy);
    auto p = policy.getProbability(environ, s0, a0);
    CHECK(p != Approx(1.0F));
  }
}

TEST_CASE("Coin MPD can undergo policy iteration") {

  auto data = CoinModelDataFixture{};
  auto &[s0, s1, a0, a1, transitionModel, environ, policy, policyState, policyAction, _v0, valueFunction, _v1] = data;

  // force the q_table to have non-optimal values
  policy.at(CoinDistributionPolicy::KeyMaker::make(environ, s0, a0)).value = 1.0F;
  policy.at(CoinDistributionPolicy::KeyMaker::make(environ, s0, a1)).value = 1.0F;
  policy.at(CoinDistributionPolicy::KeyMaker::make(environ, s1, a0)).value = 0.0F;
  policy.at(CoinDistributionPolicy::KeyMaker::make(environ, s1, a1)).value = 0.0F;
  policy_iteration(valueFunction, environ, policy, 1e-2F);
  auto p = policy.getProbability(environ, s0, a0);
  CHECK(p != Approx(1.0F));

  // policy.printQTable(environ);
}