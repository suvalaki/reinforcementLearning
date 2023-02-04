#include "catch.hpp"
#include <iostream>

#include "xtensor/xio.hpp"
#include <xtensor/xrandom.hpp>
// XTensor View:
#include "xtensor/xview.hpp"

#include "examples/blackjack.hpp"

#include "policy/epsilon_greedy_policy.hpp"
#include "policy/random_policy.hpp"

using namespace examples::blackjack;

TEST_CASE("blackjack") {

  auto state = BlackjackState();
  auto environment = BlackjackEnvironment<BlackjackReward, BlackjackReturn>();
  environment.reset();
  std::cout << "blackjack " << environment.state << "\n";
  std::cout << "value: " << environment.state.calculateHandValue<0>() << "\n";

  auto action = BlackjackAction(1);
  std::cout << "action: " << action << "\n";
  auto s2 = action.step(environment.state);
  std::cout << "blackjack " << s2 << "\n";


  // Create an epsilon  greedy policy and run the bandit over them.
  auto epsilonGreedy = policy::EpsilonGreedyPolicy<
    BlackjackEnvironment<BlackjackReward, BlackjackReturn>>{0.2F};
  // auto epsilonGreedy = policy::RandomPolicy<
  //   BlackjackEnvironment<BlackjackReward, BlackjackReturn>>{};
  std::cout << "EPSILON GREEDY ACTIONS\n";
  for (int i = 0; i < 100; i++) {
    auto recommendedAction = epsilonGreedy(environment.state);
    auto transition = environment.step(recommendedAction);
    epsilonGreedy.update(transition);
    environment.update(transition);

    // std::cout << transition.nextState << "\n";
  }

  // epsilonGreedy.printQTable();



}
