#include <iostream>

#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include <xtensor/xrandom.hpp>

#include "examples/blackjack.hpp"

#include "monte_carlo/policy_control.hpp"
#include "policy/epsilon_greedy_policy.hpp"
#include "policy/random_policy.hpp"

using namespace examples::blackjack;

int main() {
  auto state = BlackjackState();
  auto environment = BlackjackEnvironment<BlackjackReward, BlackjackReturn>();
  environment.reset();

  auto action = BlackjackAction(1);
  auto s2 = action.step(environment.state);

  // Create an epsilon  greedy policy and run the bandit over them.
  auto epsilonGreedy = policy::EpsilonGreedyPolicy<BlackjackEnvironment<BlackjackReward, BlackjackReturn>>{0.2F};
  std::cout << "EPSILON GREEDY ACTIONS\n";

  // std::cout << "Progress[]";
  for (int i = 0; i < 10; i++) {
    auto recommendedAction = epsilonGreedy(environment.state);
    auto transition = environment.step(recommendedAction);
    epsilonGreedy.update(transition);
    environment.update(transition);

    // std::cout << transition.nextState << "\n";
  }

  // Validate works with monte carlo control now that it is a finite process
  monte_carlo::monte_carlo_on_policy_first_visit_control_with_exploring_starts<10>(epsilonGreedy, environment, 15);

  epsilonGreedy.printQTable();
}