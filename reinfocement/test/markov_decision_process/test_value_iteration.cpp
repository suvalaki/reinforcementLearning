#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <iostream>
#include <limits>

#include "coin_mdp.hpp"
#include "markov_decision_process/finite_transition_model.hpp"
#include "markov_decision_process/policy_iteration.hpp"
#include "markov_decision_process/value_iteration.hpp"
#include "policy/distribution_policy.hpp"
#include "policy/random_policy.hpp"

using namespace Catch;
using namespace environment;
using namespace markov_decision_process;
using namespace markov_decision_process::value_iteration;

TEST_CASE("Coin MPD can undergo policy value iteration") {

  auto data = CoinModelDataFixture{};
  auto &[s0, s1, a0, a1, transitionModel, environ, policy, policyState, policyAction, _v0, valueFunction, _v1] = data;

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
    policy.at(CoinDistributionPolicy::KeyMaker::make(environ, s0, a0)).value = 0.0F;
    policy.at(CoinDistributionPolicy::KeyMaker::make(environ, s0, a1)).value = 0.0F;
    policy.at(CoinDistributionPolicy::KeyMaker::make(environ, s1, a0)).value = 0.0F;
    policy.at(CoinDistributionPolicy::KeyMaker::make(environ, s1, a1)).value = 0.0F;
    value_iteration::value_iteration(valueFunction, environ, policy, 1e-3F);

    policy.prettyPrint();

    // these are the optimals after 1 iteration
    auto p00 = policy.getProbability(environ, s0, a0);
    REQUIRE_THAT(p00, Catch::Matchers::WithinAbs(0.0, std::numeric_limits<float>::epsilon()) );

    auto p01 = policy.getProbability(environ, s0, a1);
    REQUIRE_THAT(p01, Catch::Matchers::WithinAbs(1.0, std::numeric_limits<float>::epsilon()) );

    auto p10 = policy.getProbability(environ, s1, a0);
    REQUIRE_THAT(p10, Catch::Matchers::WithinAbs(1.0, std::numeric_limits<float>::epsilon()) );

    auto p11 = policy.getProbability(environ, s1, a1);
    REQUIRE_THAT(p11, Catch::Matchers::WithinAbs(0.0, std::numeric_limits<float>::epsilon()) );
  }
}