#include "catch.hpp"
#include <cmath>
#include <iostream>

#include "markov_decision_process/coin_mdp.hpp"
#include "policy/finite/epsilon_greedy_policy.hpp"
#include "policy/finite/greedy_policy.hpp"
#include "policy/finite/random_policy.hpp"
#include "policy/objectives/finite_value_function_combination.hpp"
#include "temporal_difference/value_iteration.hpp"
#include "temporal_difference/value_update/double_q_learning.hpp"

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
  updater.update(epsGreedy, policy0, policy1, environ, a0, 0.5F);

  // Values all start at zero. After the reset we start at state s0. we can only have updated s0 so far...
  // We know the reward is 1.0 no matter the action or next state. So the value update is defined by:
  CHECK(((policy0.valueAt(decltype(updater)::KeyMaker::make(environ, s0, a0)) != Approx(0.0)) or
         (policy1.valueAt(decltype(updater)::KeyMaker::make(environ, s0, a0)) != Approx(0.0))));
}
