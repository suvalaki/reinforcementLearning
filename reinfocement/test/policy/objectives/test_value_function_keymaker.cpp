#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <iostream>
#include <limits>

#include "environment_fixtures.hpp"
#include "policy/objectives/value_function_keymaker.hpp"

using namespace policy;
using namespace fixtures;
using namespace policy::objectives;

TEST_CASE("StateActionKeymaker", "[policy][objectives][KeyMaker]") {

  auto env = S2A2();
  auto state = typename S2A2::StateType(1, {});
  auto action = typename S2A2::ActionSpace(1);

  auto key = StateActionKeymaker<S2A2>::make(env, state, action);
  CHECK(key.first == state);
  CHECK(key.second == action);

  auto state_from_key = StateActionKeymaker<S2A2>::get_state_from_key(env, key);
  CHECK(state_from_key == state);

  auto action_from_key = StateActionKeymaker<S2A2>::get_action_from_key(env, key);
  CHECK(action_from_key == action);
}

TEST_CASE("ActionKeymaker", "[policy][objectives][KeyMaker]") {

  auto env = S2A2();
  auto state = typename S2A2::StateType(1, {});
  auto action = typename S2A2::ActionSpace(1);

  auto key = ActionKeymaker<S2A2>::make(env, state, action);
  CHECK(key == action);

  auto state_from_key = ActionKeymaker<S2A2>::get_state_from_key(env, key);
  CHECK(state_from_key == env.getNullState()); // since state isnt available from this key

  auto action_from_key = ActionKeymaker<S2A2>::get_action_from_key(env, key);
  CHECK(action_from_key == action);
}

TEST_CASE("StateKeymaker", "[policy][objectives][KeyMaker]") {

  auto env = S2A2();
  auto state = typename S2A2::StateType(1, {});
  auto action = typename S2A2::ActionSpace(1);

  auto key = StateKeymaker<S2A2>::make(env, state, action);
  CHECK(key == state);

  auto state_from_key = StateKeymaker<S2A2>::get_state_from_key(env, key);
  CHECK(state_from_key == state);

  auto action_from_key = StateKeymaker<S2A2>::get_action_from_key(env, key);
  CHECK(action_from_key == env.getNullAction()); // since action isnt available from this key
}
