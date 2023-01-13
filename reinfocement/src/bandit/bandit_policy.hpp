#pragma once
#include <iomanip>
#include <iostream>

#include "bandit.hpp"
#include "bandit_environment.hpp"
#include "policy.hpp"

using namespace policy;

namespace bandit::policy {

template <typename ENVIRON_T, float STARTING_VALUE = 1.0F>
struct GreedyPolicy : Policy<ENVIRON_T> {

  using EnvironmentType = ENVIRON_T;
  using StateType = typename EnvironmentType::StateType;
  using ActionSpace = typename EnvironmentType::ActionSpace;
  using TransitionType = typename EnvironmentType::TransitionType;

  std::array<float, EnvironmentType::N> actionValueEstimate;
  std::array<int, EnvironmentType::N> actionSelectionCount;

  GreedyPolicy() {
    actionValueEstimate.fill(STARTING_VALUE);
    actionSelectionCount.fill(0);
  }

  // Run the policy over the current state of the environment
  // iterate over s.observableBanditSample and return the
  // argmax item. If there are multiple argmax items, return a random one. If
  // there are no argmax items, return a random item.
  ActionSpace operator()(const StateType &s) override {
    auto result = ActionSpace{};
    auto idxArgmax = std::distance(actionValueEstimate.begin(),
                                   std::max_element(actionValueEstimate.begin(),
                                                    actionValueEstimate.end()));
    // std::array<float, EnvironmentType::N> banditChoice = {false};
    result.banditChoice[idxArgmax] = true;
    return result;
  }

  void update(const TransitionType &s) override{};

  // print the current action value estimate as a table where each cell takes
  // the same space
  void printActionValueEstimate() {
    std::cout << "Action Value Estimate" << std::endl;
    std::cout << "---------------------" << std::endl;
    for (int i = 0; i < EnvironmentType::N; i++) {
      std::cout << std::setw(5) << actionValueEstimate[i] << " ";
    }
    std::cout << "\n";
    for (int i = 0; i < EnvironmentType::N; i++) {
      std::cout << std::setw(5) << actionSelectionCount[i] << " ";
    }
    std::cout << std::endl;
  }
};

} // namespace bandit::policy