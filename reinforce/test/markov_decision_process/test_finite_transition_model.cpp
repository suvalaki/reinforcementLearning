#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <iostream>

#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>

#include <reinforce/markov_decision_process/finite_transition_model.hpp>
#include <reinforce/policy/objectives/value_function_keymaker.hpp>
#include <reinforce/policy/random_policy.hpp>

#include "coin_mdp.hpp"

TEST_CASE("Finite MDP creating a type works") {

  // // Fill out the entire matrix of transition probs
  auto data = CoinModelDataFixture();
}
