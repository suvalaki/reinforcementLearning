#include "catch.hpp"
#include <iostream>

#include "coin_mdp.hpp"
#include "markov_decision_process/finite_transition_model.hpp"
#include "markov_decision_process/policy_iteration.hpp"
#include "markov_decision_process/value_iteration.hpp"
#include "policy/distribution_policy.hpp"
#include "policy/random_policy.hpp"

using namespace environment;
using namespace markov_decision_process;
using namespace markov_decision_process::value_iteration;

TEST_CASE("Coin MPD can undergo policy value iteration") {

  auto data = CoinModelDataFixture{};
  auto &[s0, s1, a0, a1, transitionModel, environ, policy, valueFunction] = data;

  SECTION("value iteration estimation step succesfully update the value") {
    // Validate that the value iteration has indeed updated the value
    auto initialValue = valueFunction.valueAt(s0);
    auto val = value_iteration_policy_estimation_step(valueFunction, environ, s0);
    valueFunction.at(s0).value = val;
    CHECK_FALSE(initialValue == valueFunction.at(s0).value);
  }

  SECTION("Complete pass over all states works") {
    // because we need to compare the values we must at least initialise them.
    // valueFunction.initialize
    // because we need to compare the values we must at least initialise them.
    valueFunction.initialize(environ);
    auto initialValues = valueFunction;
    // valueFunction.policy_evaluation(environ, policy, 1e-3F);
    value_iteration_policy_estimation(valueFunction, environ, 1e-3F);
    for (auto &[state, value] : valueFunction) {
      CHECK(value != initialValues.at(state).value);
    }
  }

  SECTION("Full Value Iteration function works") {
    // force the q_table to have non-optimal values
    policy.at(CoinDistributionPolicy::KeyMaker::make(s0, a0)).value = 0.0F;
    policy.at(CoinDistributionPolicy::KeyMaker::make(s0, a1)).value = 0.0F;
    policy.at(CoinDistributionPolicy::KeyMaker::make(s1, a0)).value = 0.0F;
    policy.at(CoinDistributionPolicy::KeyMaker::make(s1, a1)).value = 0.0F;
    value_iteration::value_iteration(valueFunction, environ, policy, 1e-3F);

    // these are the optimals after 1 iteration
    auto p00 = policy.getProbability(environ, s0, CoinDistributionPolicy::KeyMaker::make(s0, a0));
    CHECK(p00 == Approx(0.0F));

    auto p01 = policy.getProbability(environ, s0, CoinDistributionPolicy::KeyMaker::make(s0, a1));
    CHECK(p01 == Approx(1.0F));

    auto p10 = policy.getProbability(environ, s0, CoinDistributionPolicy::KeyMaker::make(s1, a0));
    CHECK(p10 == Approx(1.0F));

    auto p11 = policy.getProbability(environ, s0, CoinDistributionPolicy::KeyMaker::make(s1, a1));
    CHECK(p11 == Approx(0.0F));
  }
}