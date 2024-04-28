#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <iostream>
#include <limits>

#include <reinforce/policy/finite/random_policy.hpp>

#include "environment_fixtures.hpp"

using namespace Catch;
using namespace policy;
using namespace fixtures;

TEST_CASE("FiniteRandomPolicy", "[policy][finite][random]") {

  // The finite random policy doesnt have a value function associated with it. Instead it has a uniform distribution
  // over the state-actions

  SECTION("Simple environment where all actions are reachable from all states") {

    auto env = S2A2{};
    auto policy = FiniteRandomPolicy<S2A2>{};

    // operator(), sampleAction always return a random action
    CHECK_THROWS(policy.getArgmaxAction(env, env.stateFromIndex(0)));

    // prob is the size of the number of actions we can reach - testing state set size isnt impacting result.
    REQUIRE(policy.getProbability(env, env.stateFromIndex(0), env.actionFromIndex(1)) == Approx(0.5));
    REQUIRE(policy.getProbability(env, env.stateFromIndex(0), env.actionFromIndex(0)) == Approx(0.5));

    REQUIRE(policy.getLogProbability(env, env.stateFromIndex(0), env.actionFromIndex(1)) == Approx(std::log(0.5F)));
    REQUIRE(policy.getLogProbability(env, env.stateFromIndex(0), env.actionFromIndex(0)) == Approx(std::log(0.5F)));
  }

  SECTION("Transition environment when we have a different mechanism for mapping states to actions") {

    auto env = MS5A10{};
    auto policy = FiniteRandomPolicy<MS5A10>{};

    // // operator(), sampleAction always return a random action
    CHECK_THROWS(policy.getArgmaxAction(env, env.stateFromIndex(0)));

    // our env only allows for actions up to and including the index of the state
    // Hence the probability of chosing an action from a state should be the state index as well.
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
  }
}
