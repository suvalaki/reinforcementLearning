#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <limits>

#include <reinforce/policy/finite/greedy_policy.hpp>
#include <reinforce/policy/objectives/finite_value_function.hpp>
#include <reinforce/policy/objectives/finite_value_function_combination.hpp>
#include <reinforce/policy/objectives/value_function_combination.hpp>
#include <reinforce/policy/objectives/value_function_keymaker.hpp>

#include "environment_fixtures.hpp"

using namespace Catch;
using namespace policy;
using namespace fixtures;
using namespace policy::objectives;

TEST_CASE("AdditiveFiniteValueFunctionCombination", "[policy][objectives][combination]") {

  auto env = S1A2{};
  auto policy0 = FiniteGreedyPolicyC<S1A2, StateActionKeymaker>{};
  auto policy1 = FiniteGreedyPolicyC<S1A2, StateActionKeymaker>{};
  using policyT = decltype(policy0);
  using AdditiveType = AdditiveFiniteValueFunctionCombination<decltype(policy0), decltype(policy1)>;
  auto combination = AdditiveType(policy0, policy1);

  combination.initialize(env);
  policy0.initialize(env);

  auto key = policyT::KeyMaker::make(env, env.stateFromIndex(0), env.actionFromIndex(1));
  REQUIRE(combination(key).value == Approx(0.0));
  policy0[key].value = 2.0;
  REQUIRE(combination(key).value == Approx(2.0));
  policy1[key].value = 1.0;
  REQUIRE(combination(key).value == Approx(3.0));

  auto policy2 = FiniteGreedyPolicy<AdditiveType>(policy0, policy1);
  REQUIRE(policy2.getValue(env, env.stateFromIndex(0), env.actionFromIndex(1)) == Approx(3.0));
}

TEST_CASE("AdditiveFiniteValueFunctionCombination::getArgmaxKey", "[policy][objectives][combination]") {

  // Need to validate that the argmax returned is indeed the from the sum of the keys correctly

  auto env = S1A2{};
  auto policy0 = FiniteGreedyPolicyC<S1A2, StateActionKeymaker>{};
  auto policy1 = FiniteGreedyPolicyC<S1A2, StateActionKeymaker>{};
  using policyT = decltype(policy0);
  auto combination = AdditiveFiniteValueFunctionCombination<decltype(policy0), decltype(policy1)>(policy0, policy1);

  combination.initialize(env);

  auto key = policyT::KeyMaker::make(env, env.stateFromIndex(0), env.actionFromIndex(1));
  policy0[key].value = 2.0;
  policy1[key].value = 1.0;
  REQUIRE(combination.getArgmaxKey(env, env.stateFromIndex(0)) == key);
}

TEST_CASE("AdditiveFiniteValueFunctionCombination_can_be_used_for_other_policy", "[policy][objectives][combination]") {

  auto env = S1A2{};
  auto policy0 = FiniteGreedyPolicyC<S1A2, StateActionKeymaker>{};
  auto policy1 = FiniteGreedyPolicyC<S1A2, StateActionKeymaker>{};
  using policyT = decltype(policy0);
  using AdditiveType = AdditiveFiniteValueFunctionCombination<decltype(policy0), decltype(policy1)>;
  auto combination = AdditiveType(policy0, policy1);

  combination.initialize(env);
  policy0.initialize(env);

  // Validate we can make a greedy policy based on this value function
  static_assert(policy::objectives::isFiniteAdditiveValueFunctionCombination<AdditiveType>);
  // auto policy2 = FiniteGreedyPolicy<AdditiveType>{};
  auto policy2 = FiniteGreedyPolicy<AdditiveType>(policy0, policy1);
  auto policy3 = FiniteGreedyPolicy<AdditiveType>(combination);

  // We should be able to impact what is chosen by these new greedy policies by modifying the underlying
  // value functions inside policy0 and policy1. We can make a new one the max.
  auto key = policyT::KeyMaker::make(env, env.stateFromIndex(0), env.actionFromIndex(1));
  policy0[key].value = 2.0;
  auto p2Action0 = policy2(env, env.stateFromIndex(0));
  auto p3Action0 = policy3(env, env.stateFromIndex(0));
  REQUIRE(p2Action0 == env.actionFromIndex(1));
  REQUIRE(p3Action0 == env.actionFromIndex(1));

  // Adding some value to the other policy should change the action when the value is greater.
  auto key1 = policyT::KeyMaker::make(env, env.stateFromIndex(0), env.actionFromIndex(0));
  policy1[key1].value = 3.0;
  policy2.prettyPrint();
  auto p2Action1 = policy2(env, env.stateFromIndex(0));
  auto p3Action1 = policy3(env, env.stateFromIndex(0));
  CHECK(p2Action1 == env.actionFromIndex(0));
  CHECK(p3Action1 == env.actionFromIndex(0));

  // Continuing to add to policy1 on action1 enough such that each individually is less by the sum is
  // greater changes the answer back.
  policy1[key].value = 2.0;
  auto p2Action2 = policy2(env, env.stateFromIndex(0));
  auto p3Action2 = policy3(env, env.stateFromIndex(0));
  CHECK(p2Action2 == env.actionFromIndex(1));
  CHECK(p3Action2 == env.actionFromIndex(1));
}