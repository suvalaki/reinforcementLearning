#include "catch.hpp"
#include <iostream>
#include <limits>

#include "environment_fixtures.hpp"
#include "policy/finite/greedy_policy.hpp"
#include "policy/objectives/value_function_keymaker.hpp"

using namespace policy;
using namespace fixtures;
using namespace policy::objectives;

TEST_CASE("FiniteGreedyPolicy", "[policy][finite][greedy]") {

  auto env = S1A4{};
  auto policy = FiniteGreedyPolicyC<S1A4, StateActionKeymaker>{};

  // set the value for action1 and validate thhat the policy returns action1
  {
    // Note: the max we are setting is in the middle of the pack.
    policy[decltype(policy)::KeyMaker::make(env, env.stateFromIndex(0), env.actionFromIndex(1))].value = 2.0;
    auto recommendedAction = policy(env, env.stateFromIndex(0));
    REQUIRE(recommendedAction == env.actionFromIndex(1));
    auto sample = policy.sampleAction(env, env.stateFromIndex(0));
    REQUIRE(sample == recommendedAction); // samples from the greedy always
    auto argmax = policy.getArgmaxAction(env, env.stateFromIndex(0));
    REQUIRE(argmax == recommendedAction); // argmax is the same as the greedy

    REQUIRE(policy.getProbability(env, env.stateFromIndex(0), env.actionFromIndex(0)) == Approx(0.0));
    REQUIRE(policy.getProbability(env, env.stateFromIndex(0), env.actionFromIndex(1)) == Approx(1.0));
    REQUIRE(policy.getProbability(env, env.stateFromIndex(0), env.actionFromIndex(2)) == Approx(0.0));
    REQUIRE(policy.getProbability(env, env.stateFromIndex(0), env.actionFromIndex(3)) == Approx(0.0));

    REQUIRE(policy.getLogProbability(env, env.stateFromIndex(0), env.actionFromIndex(1)) == 0.0);
    REQUIRE(
        policy.getLogProbability(env, env.stateFromIndex(0), env.actionFromIndex(0)) ==
        -std::numeric_limits<float>::infinity());
  }

  // Change the value for action0 to be above action1 and validate that the policy returns action0
  {
    policy[decltype(policy)::KeyMaker::make(env, env.stateFromIndex(0), env.actionFromIndex(0))].value = 3.0;
    auto recommendedAction = policy(env, env.stateFromIndex(0));
    REQUIRE(recommendedAction == env.actionFromIndex(0));
    auto sample = policy.sampleAction(env, env.stateFromIndex(0));
    REQUIRE(sample == recommendedAction); // samples from the greedy always
    auto argmax = policy.getArgmaxAction(env, env.stateFromIndex(0));
    REQUIRE(argmax == recommendedAction); // argmax is the same as the greedy

    REQUIRE(policy.getProbability(env, env.stateFromIndex(0), env.actionFromIndex(1)) == Approx(0.0));
    REQUIRE(policy.getProbability(env, env.stateFromIndex(0), env.actionFromIndex(0)) == Approx(1.0));

    REQUIRE(
        policy.getLogProbability(env, env.stateFromIndex(0), env.actionFromIndex(1)) ==
        -std::numeric_limits<float>::infinity());
    REQUIRE(policy.getLogProbability(env, env.stateFromIndex(0), env.actionFromIndex(0)) == 0.0);
  }
}