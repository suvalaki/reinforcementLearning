#include "bandit.hpp"
#include "explore.hpp"
#include <iomanip>
#include <iostream>
#include <utility>

int main(int argc, char *argv[]) {
  char *tmp;
  // Number of bandits
  std::size_t n_bandits = static_cast<std::size_t>(strtol(argv[1], &tmp, 10));
  // Number of samples to take
  std::size_t n_samples = static_cast<std::size_t>(strtol(argv[2], &tmp, 10));

  // Starting estimate for each banits value
  double baseValue = 0;
  if (argc >= 4) {
    baseValue = atof(argv[3]);
  }

  // setup bandits
  std::minstd_rand generator = {};
  NormalPriorNormalMultiArmedBandit<double> bandits(n_bandits, generator);
  std::vector<double> initialValueEstimates;
  initialValueEstimates.resize(n_bandits);
  std::fill(initialValueEstimates.begin(), initialValueEstimates.end(),
            baseValue);
  double eps = 0.1;
  strategy::GreedyStrategy<double> greedyStrategy(initialValueEstimates);
  // strategy::EpsilonGreedyStrategy<double> epsilonGreedyStrategy(
  //    generator, initialValueEstimates, eps);

  std::vector<std::pair<std::string, strategy::StrategyBase<double> &>>
      strategyVec;

  strategyVec.emplace_back(std::string("greedy"), greedyStrategy);
  // strategyVec.emplace_back(std::string("epsilon-greedy"),
  //                         epsilonGreedyStrategy);

  std::cout << "Generatimng Ranom Numbers"
            << " with " << n_bandits << " bandits and " << n_samples
            << " samples\n";
  for (size_t i = 0; i < n_samples; i++) {
    if (i == 0) {
      for (size_t j = 0; j < n_bandits; j++) {
        std::cout << std::setw(9) << "b" << j << ", ";
      }
      for (auto &[name, strategy] : strategyVec) {
        std::cout << std::setw(9) << name << "_action, ";
        std::cout << std::setw(9) << name << "_actionVal, ";
        std::cout << std::setw(9) << name << "_actionEst, ";
        std::cout << std::setw(9) << name << "_gredyEst, ";
      }
      std::cout << "\n";

    } else {
      std::cout << "\n";
    }
    std::vector<double> result = bandits.sample(generator);
    for (auto const &v : result) {
      std::cout << std::setw(9) << v << ", ";
    }

    for (auto &[name, strategy] : strategyVec) {
      // std::size_t action = strategy.step();
      std::size_t action = strategy.exploit();
      double actionValue = result[action];
      strategy.update(action, actionValue);
      double newActionValueEst = strategy.getActionValueEstimate(action);
      double bestActionEst =
          strategy.getActionValueEstimate(strategy.exploit());
      std::cout << std::setw(3) << action << ", " << std::setw(9) << actionValue
                << ", " << std::setw(9) << newActionValueEst << ", "
                << std::setw(9) << bestActionEst << ", ";
    }
  }
  std::cout << "\n\nFinal Action Value Estimates";
  std::cout << "\n";

  std::cout << std::setw(10) << "actual\t";
  auto actualValues = bandits.getBanditValues();
  for (auto &v : actualValues) {
    std::cout << std::setw(10) << v << "\t";
  }
  std::cout << "\n";

  for (auto &[name, strategy] : strategyVec) {
    auto actionValues = strategy.getActionValueEstimate();
    std::cout << std::setw(10) << name << "\t";
    for (auto const &v : actionValues) {
      std::cout << std::setw(10) << v << "\t";
    }
    std::cout << "\n";
  }

  return 0;
}