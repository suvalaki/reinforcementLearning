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

int main() {
  auto state = BlackjackState();
  using EnvT = BlackjackEnvironment<BlackjackReward, BlackjackReturn>;
  auto environment = EnvT();
  environment.reset();

  auto action = BlackjackAction(1);
  auto s2 = action.step(environment.state);

  // Create an epsilon  greedy policy and run the bandit over them.
  using BlackjackRandom = policy::FiniteRandomPolicy<EnvT>;
  using BlackjackKeyMaker = policy::objectives::StateActionKeymaker<EnvT>;
  using BlackjackValue = policy::objectives::FiniteValue<EnvT>;
  using BlackjackValueFunction =
      policy::objectives::FiniteValueFunction<policy::objectives::ValueFunction<BlackjackKeyMaker, BlackjackValue>>;
  using BlackjackGreedy = policy::FiniteGreedyPolicy<BlackjackValueFunction>;
  auto explorPolicy = BlackjackRandom();
  auto greedyPolicy = BlackjackGreedy();
  static_assert(policy::isFinitePolicyValueFunctionMixin<BlackjackGreedy>);
  auto epsilonGreedy = policy::FiniteEpsilonGreedyPolicy<BlackjackRandom, BlackjackGreedy>{explorPolicy, {}, 0.2F};
  std::cout << "EPSILON GREEDY ACTIONS\n";

  // std::cout << "Progress[]";
  for (int i = 0; i < 1000; i++) {
    auto recommendedAction = epsilonGreedy(environment, environment.state);
    auto transition = environment.step(recommendedAction);
    epsilonGreedy.update(environment, transition);
    environment.update(transition);

    auto z0 = epsilonGreedy.getProbability(environment, environment.state, recommendedAction);
    // auto z = epsilonGreedy.getProbability(environment, environment.state, recommendedAction);

    // std::cout << transition.nextState << "\n";
  }

  // Validate works with monte carlo control now that it is a finite process
  monte_carlo::monte_carlo_on_policy_first_visit_control_with_exploring_starts<10>(epsilonGreedy, environment, 15);

  epsilonGreedy.prettyPrint();
}