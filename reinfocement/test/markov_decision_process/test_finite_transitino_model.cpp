#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <iostream>

#include "xtensor/xio.hpp"
#include <xtensor/xrandom.hpp>
// XTensor View:
#include "xtensor/xview.hpp"

#include "markov_decision_process/finite_transition_model.hpp"
#include "policy/objectives/value_function_keymaker.hpp"
#include "policy/random_policy.hpp"

#include "coin_mdp.hpp"

TEST_CASE("Finite MDP creating a type works") {

  // // Fill out the entire matrix of transition probs
  auto data = CoinModelDataFixture();
}
