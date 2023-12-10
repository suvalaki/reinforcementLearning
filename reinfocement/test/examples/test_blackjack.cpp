#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <iostream>

#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include <xtensor/xrandom.hpp>

#include "examples/blackjack.hpp"

#include "monte_carlo/policy_control.hpp"
#include "policy/finite/epsilon_greedy_policy.hpp"
#include "policy/finite/greedy_policy.hpp"
#include "policy/finite/random_policy.hpp"
#include "policy/objectives/value_function_keymaker.hpp"

using namespace examples::blackjack;

TEST_CASE("blackjack") {

  auto state = BlackjackState();
  using EnvT = BlackjackEnvironment<BlackjackReward, BlackjackReturn>;
  auto environment = EnvT();
  environment.reset();

  auto action = BlackjackAction(1);
  auto s2 = action.step(environment.state);

  // Create an epsilon  greedy policy and run the bandit over them.
  using BlackjackKeyMaker = policy::objectives::StateActionKeymaker<EnvT>;
  using BlackjackRandom = policy::FiniteRandomPolicy<EnvT>;
  using BlackjackValue = policy::objectives::FiniteValue<EnvT>;
  using BlackjackValueFunction =
      policy::objectives::FiniteValueFunction<policy::objectives::ValueFunction<BlackjackKeyMaker, BlackjackValue>>;
  using BlackjackGreedy = policy::FiniteGreedyPolicy<BlackjackValueFunction>;
  auto explorPolicy = BlackjackRandom();
  auto epsilonGreedy = policy::FiniteEpsilonGreedyPolicy<BlackjackRandom, BlackjackGreedy>{explorPolicy, {}, 0.2F};
  // std::cout << "EPSILON GREEDY ACTIONS\n";
  for (int i = 0; i < 10; i++) {
    auto recommendedAction = epsilonGreedy(environment, environment.state);
    auto transition = environment.step(recommendedAction);
    epsilonGreedy.update(environment, transition);
    environment.update(transition);

    // std::cout << transition.nextState << "\n";
  }

  // Validate works with monte carlo control now that it is a finite process
  monte_carlo::monte_carlo_on_policy_first_visit_control_with_exploring_starts<10>(epsilonGreedy, environment, 15);

  // epsilonGreedy.printQTable();
}
