#include "catch.hpp"
#include <iostream>

#include "xtensor/xio.hpp"
#include <xtensor/xrandom.hpp>
// XTensor View:
#include "xtensor/xview.hpp"

#include "markov_decision_process/finite_transition_model.hpp"
#include "policy/random_policy.hpp"
#include "policy/state_action_keymaker.hpp"

#include "coin_mdp.hpp"

TEST_CASE("Finite MDP creating a type works") {

  // // Fill out the entire matrix of transition probs
  auto data = CoinModelDataFixture();
}
