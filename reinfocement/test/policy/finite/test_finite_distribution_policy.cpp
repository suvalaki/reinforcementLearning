#include "catch.hpp"
#include <cmath>
#include <iostream>
#include <limits>

#include "environment_fixtures.hpp"
#include "policy/finite/distribution_policy.hpp"
#include "policy/objectives/value_function_keymaker.hpp"

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
  CHECK(policy.valueAt(env, env.stateFromIndex(0), env.actionFromIndex(0)) == Approx(policy.initial_value));
  CHECK(policy.valueAt(env, env.stateFromIndex(1), env.actionFromIndex(0)) == Approx(policy.initial_value));
  CHECK(policy.valueAt(env, env.stateFromIndex(2), env.actionFromIndex(0)) == Approx(policy.initial_value));
  CHECK(policy.valueAt(env, env.stateFromIndex(3), env.actionFromIndex(0)) == Approx(policy.initial_value));
  CHECK(policy.valueAt(env, env.stateFromIndex(4), env.actionFromIndex(0)) == Approx(policy.initial_value));

  CHECK(policy.getProbability(env, env.stateFromIndex(0), env.actionFromIndex(0)) == Approx(1.0));
  CHECK(policy.getProbability(env, env.stateFromIndex(1), env.actionFromIndex(0)) == Approx(1 / 2.0F));
  CHECK(policy.getProbability(env, env.stateFromIndex(2), env.actionFromIndex(0)) == Approx(1 / 3.0F));
  CHECK(policy.getProbability(env, env.stateFromIndex(3), env.actionFromIndex(0)) == Approx(1 / 4.0F));
  CHECK(policy.getProbability(env, env.stateFromIndex(4), env.actionFromIndex(0)) == Approx(1 / 5.0F));

  CHECK(policy.getLogProbability(env, env.stateFromIndex(0), env.actionFromIndex(0)) == Approx(std::log(1.0)));
  CHECK(policy.getLogProbability(env, env.stateFromIndex(1), env.actionFromIndex(0)) == Approx(std::log(1 / 2.0F)));
  CHECK(policy.getLogProbability(env, env.stateFromIndex(2), env.actionFromIndex(0)) == Approx(std::log(1 / 3.0F)));
  CHECK(policy.getLogProbability(env, env.stateFromIndex(3), env.actionFromIndex(0)) == Approx(std::log(1 / 4.0F)));
  CHECK(policy.getLogProbability(env, env.stateFromIndex(4), env.actionFromIndex(0)) == Approx(std::log(1 / 5.0F)));

  CHECK(policy.getNormalisationConstant(env, env.stateFromIndex(0)) == 1);
  CHECK(policy.getNormalisationConstant(env, env.stateFromIndex(1)) == 2);
  CHECK(policy.getNormalisationConstant(env, env.stateFromIndex(2)) == 3);
  CHECK(policy.getNormalisationConstant(env, env.stateFromIndex(3)) == 4);
  CHECK(policy.getNormalisationConstant(env, env.stateFromIndex(4)) == 5);

  // Setting the determinist policy works for a single state

  policy.setDeterministicPolicy(env, env.stateFromIndex(4), env.actionFromIndex(0));
  CHECK(policy.valueAt(env, env.stateFromIndex(0), env.actionFromIndex(0)) == Approx(policy.initial_value));
  CHECK(policy.valueAt(env, env.stateFromIndex(1), env.actionFromIndex(0)) == Approx(policy.initial_value));
  CHECK(policy.valueAt(env, env.stateFromIndex(2), env.actionFromIndex(0)) == Approx(policy.initial_value));
  CHECK(policy.valueAt(env, env.stateFromIndex(3), env.actionFromIndex(0)) == Approx(policy.initial_value));

  CHECK(policy.valueAt(env, env.stateFromIndex(4), env.actionFromIndex(0)) == Approx(policy.max_policy_value));
  CHECK(policy.valueAt(env, env.stateFromIndex(4), env.actionFromIndex(1)) == Approx(policy.min_policy_value));
  CHECK(policy.valueAt(env, env.stateFromIndex(4), env.actionFromIndex(2)) == Approx(policy.min_policy_value));
  CHECK(policy.valueAt(env, env.stateFromIndex(4), env.actionFromIndex(3)) == Approx(policy.min_policy_value));

  CHECK(policy.getProbability(env, env.stateFromIndex(4), env.actionFromIndex(0)) == Approx(1.0));
  for (int i = 1; i < 10; i++) {
    CHECK(policy.getProbability(env, env.stateFromIndex(4), env.actionFromIndex(i)) == Approx(0.0));
  }

  // size of probabilities should be the size of available actions
  for (int i = 0; i < 5; i++) {
    CHECK(
        policy.getProbabilities(env, env.stateFromIndex(i)).size() ==
        env.getReachableActions(env.stateFromIndex(i)).size());
  }
}
