#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <iostream>

#include <reinforce/policy/finite/epsilon_greedy_policy.hpp>
#include <reinforce/policy/finite/greedy_policy.hpp>
#include <reinforce/policy/finite/random_policy.hpp>
#include <reinforce/policy/objectives/finite_value_function_combination.hpp>
#include <reinforce/temporal_difference/value_iteration.hpp>
#include <reinforce/temporal_difference/value_update/double_q_learning.hpp>

#include "markov_decision_process/coin_mdp.hpp"

using namespace Catch;
using namespace temporal_difference;

TEST_CASE("temporal_difference::DoubleQLearningUpdater") {

  // Because the coin MDP has a fixed reward then the updated reward will always be that fixed reward. 1.0
  auto data = CoinModelDataFixture{};
  auto &[s0, s1, a0, a1, transitionModel, environ, policySA, policyState, policyAction, valueFunction, _v1, _v2] = data;
  // auto returns = monte_carlo::n_visit_returns_initialisation(valueFunction, environ);

  auto policy0 = std::decay_t<decltype(policySA)>();
  auto policy1 = std::decay_t<decltype(policySA)>();
  auto combinedValue = policy::objectives::AdditiveFiniteValueFunctionCombination<decltype(policy0), decltype(policy1)>{
      policy0, policy1};
  auto randomPolicy = policy::FiniteRandomPolicy<std::decay_t<decltype(environ)>>();
  using CombinedGreedyPolicy = policy::FiniteGreedyPolicy<std::decay_t<decltype(combinedValue)>>;
  auto combinedGreedyPolicy = CombinedGreedyPolicy{combinedValue};
  auto epsGreedy = policy::FiniteEpsilonGreedyPolicy<std::decay_t<decltype(randomPolicy)>, CombinedGreedyPolicy>{
      randomPolicy, combinedGreedyPolicy, 0.5F};
  auto updater = DoubleQLearningUpdater<std::decay_t<decltype(epsGreedy)>>();

  policy0.initialize(environ);
  policy1.initialize(environ);

  const auto key = decltype(updater)::KeyMaker::make(environ, s0, a0);
  updater.updatePolicy(environ, policy0, policy1, key, 1.0F, 0.5F);

  // We update the off Policy with a reward of 1.0F
  CHECK((not(policy1.valueAt(key) == Approx(0.0))));

  // Argmax for Q2 is now a0
  // Lets update Q1 using this action after specifying a value for Q1 at a0
  policy0[key] = 2.0F;
  updater.updatePolicy(environ, policy1, policy0, key, 1.0F, 0.5F);
  // Q1[s,a] <- Q1[s,a] + alpha * (r + gamma * Q2[s',argmax_a(Q1[s',a])] - Q1[s,a])
  CHECK((policy0.valueAt(key) == Approx(2.0F + (1.0F + 0.5F * 2.0F - 2.0F))));
}
