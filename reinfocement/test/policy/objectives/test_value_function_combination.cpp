#include "catch.hpp"
#include <iostream>
#include <limits>

#include "environment_fixtures.hpp"
#include "policy/finite/greedy_policy.hpp"
#include "policy/objectives/finite_value_function.hpp"
#include "policy/objectives/finite_value_function_combination.hpp"
#include "policy/objectives/value_function_combination.hpp"
#include "policy/objectives/value_function_keymaker.hpp"

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

  // Validate we can make a greedy policy based on this value function
  static_assert(policy::objectives::isFiniteAdditiveValueFunctionCombination<AdditiveType>);
  // auto policy2 = FiniteGreedyPolicy<AdditiveType>{};
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