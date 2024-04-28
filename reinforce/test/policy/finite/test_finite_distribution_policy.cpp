#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <iostream>
#include <limits>

#include <reinforce/policy/finite/distribution_policy.hpp>
#include <reinforce/policy/objectives/value_function_keymaker.hpp>

#include "environment_fixtures.hpp"

using namespace Catch;
using namespace policy;
using namespace fixtures;
using namespace policy::objectives;

TEST_CASE("FiniteDistributionPolicy", "[policy][finite][distribution]") {

  // The finite distribution policy creates a distribtion over the state-actions given values.
  // It can also be set deterministicly to a specific action for a given state.

  auto env = MS5A10{};
  auto policy = FiniteDistributionPolicyC<MS5A10, StateActionKeymaker>{};
  policy.initialize(env);

  // On initialisation we should have a uniform probability to all reachable actions from this state
  REQUIRE_THAT(
      policy.valueAt(env, env.stateFromIndex(0), env.actionFromIndex(0)),
      Catch::Matchers::WithinAbs(policy.initial_value, std::numeric_limits<float>::epsilon()));
  REQUIRE_THAT(
      policy.valueAt(env, env.stateFromIndex(1), env.actionFromIndex(0)),
      Catch::Matchers::WithinAbs(policy.initial_value, std::numeric_limits<float>::epsilon()));
  REQUIRE_THAT(
      policy.valueAt(env, env.stateFromIndex(2), env.actionFromIndex(0)),
      Catch::Matchers::WithinAbs(policy.initial_value, std::numeric_limits<float>::epsilon()));
  REQUIRE_THAT(
      policy.valueAt(env, env.stateFromIndex(3), env.actionFromIndex(0)),
      Catch::Matchers::WithinAbs(policy.initial_value, std::numeric_limits<float>::epsilon()));
  REQUIRE_THAT(
      policy.valueAt(env, env.stateFromIndex(4), env.actionFromIndex(0)),
      Catch::Matchers::WithinAbs(policy.initial_value, std::numeric_limits<float>::epsilon()));

  REQUIRE_THAT(
      policy.getProbability(env, env.stateFromIndex(0), env.actionFromIndex(0)),
      Catch::Matchers::WithinAbs(1.0F, std::numeric_limits<float>::epsilon()));
  REQUIRE_THAT(
      policy.getProbability(env, env.stateFromIndex(1), env.actionFromIndex(0)),
      Catch::Matchers::WithinAbs(1 / 2.0F, std::numeric_limits<float>::epsilon()));
  REQUIRE_THAT(
      policy.getProbability(env, env.stateFromIndex(2), env.actionFromIndex(0)),
      Catch::Matchers::WithinAbs(1 / 3.0F, std::numeric_limits<float>::epsilon()));
  REQUIRE_THAT(
      policy.getProbability(env, env.stateFromIndex(3), env.actionFromIndex(0)),
      Catch::Matchers::WithinAbs(1 / 4.0F, std::numeric_limits<float>::epsilon()));
  REQUIRE_THAT(
      policy.getProbability(env, env.stateFromIndex(4), env.actionFromIndex(0)),
      Catch::Matchers::WithinAbs(1 / 5.0F, std::numeric_limits<float>::epsilon()));

  REQUIRE_THAT(
      policy.getLogProbability(env, env.stateFromIndex(0), env.actionFromIndex(0)),
      Catch::Matchers::WithinAbs(std::log(1.0F), std::numeric_limits<float>::epsilon()));
  REQUIRE_THAT(
      policy.getLogProbability(env, env.stateFromIndex(1), env.actionFromIndex(0)),
      Catch::Matchers::WithinAbs(std::log(1 / 2.0F), std::numeric_limits<float>::epsilon()));
  REQUIRE_THAT(
      policy.getLogProbability(env, env.stateFromIndex(2), env.actionFromIndex(0)),
      Catch::Matchers::WithinAbs(std::log(1 / 3.0F), std::numeric_limits<float>::epsilon()));
  REQUIRE_THAT(
      policy.getLogProbability(env, env.stateFromIndex(3), env.actionFromIndex(0)),
      Catch::Matchers::WithinAbs(std::log(1 / 4.0F), std::numeric_limits<float>::epsilon()));
  REQUIRE_THAT(
      policy.getLogProbability(env, env.stateFromIndex(4), env.actionFromIndex(0)),
      Catch::Matchers::WithinAbs(std::log(1 / 5.0F), std::numeric_limits<float>::epsilon()));

  CHECK(policy.getNormalisationConstant(env, env.stateFromIndex(0)) == 1);
  CHECK(policy.getNormalisationConstant(env, env.stateFromIndex(1)) == 2);
  CHECK(policy.getNormalisationConstant(env, env.stateFromIndex(2)) == 3);
  CHECK(policy.getNormalisationConstant(env, env.stateFromIndex(3)) == 4);
  CHECK(policy.getNormalisationConstant(env, env.stateFromIndex(4)) == 5);

  // Setting the determinist policy works for a single state

  policy.setDeterministicPolicy(env, env.stateFromIndex(4), env.actionFromIndex(0));
  REQUIRE_THAT(
      policy.valueAt(env, env.stateFromIndex(0), env.actionFromIndex(0)),
      Catch::Matchers::WithinAbs(policy.initial_value, std::numeric_limits<float>::epsilon()));
  REQUIRE_THAT(
      policy.valueAt(env, env.stateFromIndex(1), env.actionFromIndex(0)),
      Catch::Matchers::WithinAbs(policy.initial_value, std::numeric_limits<float>::epsilon()));
  REQUIRE_THAT(
      policy.valueAt(env, env.stateFromIndex(2), env.actionFromIndex(0)),
      Catch::Matchers::WithinAbs(policy.initial_value, std::numeric_limits<float>::epsilon()));
  REQUIRE_THAT(
      policy.valueAt(env, env.stateFromIndex(3), env.actionFromIndex(0)),
      Catch::Matchers::WithinAbs(policy.initial_value, std::numeric_limits<float>::epsilon()));

  REQUIRE_THAT(
      policy.valueAt(env, env.stateFromIndex(4), env.actionFromIndex(0)),
      Catch::Matchers::WithinAbs(policy.max_policy_value, std::numeric_limits<float>::epsilon()));
  REQUIRE_THAT(
      policy.valueAt(env, env.stateFromIndex(4), env.actionFromIndex(1)),
      Catch::Matchers::WithinAbs(policy.min_policy_value, std::numeric_limits<float>::epsilon()));
  REQUIRE_THAT(
      policy.valueAt(env, env.stateFromIndex(4), env.actionFromIndex(2)),
      Catch::Matchers::WithinAbs(policy.min_policy_value, std::numeric_limits<float>::epsilon()));
  REQUIRE_THAT(
      policy.valueAt(env, env.stateFromIndex(4), env.actionFromIndex(3)),
      Catch::Matchers::WithinAbs(policy.min_policy_value, std::numeric_limits<float>::epsilon()));

  REQUIRE_THAT(
      policy.getProbability(env, env.stateFromIndex(4), env.actionFromIndex(0)),
      Catch::Matchers::WithinAbs(1.0F, std::numeric_limits<float>::epsilon()));
  for (int i = 1; i < 10; i++) {
    REQUIRE_THAT(
        policy.getProbability(env, env.stateFromIndex(4), env.actionFromIndex(i)),
        Catch::Matchers::WithinAbs(0.0F, std::numeric_limits<float>::epsilon()));
  }

  // size of probabilities should be the size of available actions
  for (int i = 0; i < 5; i++) {
    CHECK(
        policy.getProbabilities(env, env.stateFromIndex(i)).size() ==
        env.getReachableActions(env.stateFromIndex(i)).size());
  }
}
