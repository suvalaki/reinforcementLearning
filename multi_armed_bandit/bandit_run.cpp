#include "bandit.hpp"
#include "strategy.hpp"
#include <iomanip>
#include <iostream>

int main(int argc, char *argv[]) {
  char *tmp;
  std::size_t n_bandits = static_cast<std::size_t>(strtol(argv[1], &tmp, 10));
  std::size_t n_samples = static_cast<std::size_t>(strtol(argv[2], &tmp, 10));

  // setup bandits
  std::minstd_rand generator = {};
  NormalPriorNormalMultiArmedBandit<double> bandits(n_bandits, generator);
  std::vector<double> initialValueEstimates;
  initialValueEstimates.resize(n_bandits);
  std::fill(initialValueEstimates.begin(), initialValueEstimates.end(), 10);
  double eps = 0.1;
  strategies::GreedyStrategy<double> greedyStrategy(bandits,
                                                    initialValueEstimates);
  strategies::EpsilonGreedyStrategy<double> epsilonGreedyStrategy(
      bandits, generator, initialValueEstimates, eps);

  std::vector<std::pair<std::string, strategies::ActionValueStrategy<double> &>>
      strategyVec;

  strategyVec.emplace_back(std::string("greedy"), greedyStrategy);
  strategyVec.emplace_back(std::string("epsilon-greedy"),
                           epsilonGreedyStrategy);

  for (size_t i = 0; i < n_samples; i++) {
    if (i == 0) {
      std::cout << "Generatimng Ranom Numbers"
                << " with " << n_bandits << " bandits and " << n_samples
                << " samples\n";
    } else {
      std::cout << "\n";
    }
    std::vector<double> result = bandits.sample(generator);
    for (auto const &v : result) {
      std::cout << std::setw(9) << v << " ";
    }

    for (auto &[name, strategy] : strategyVec) {
      std::size_t action = strategy.getAction();
      double actionValue = result[action];
      strategy.update(actionValue, action);
      double newActionValueEst = strategy.estimatedActionValue(action);
      std::cout << name << ": (" << action << ", " << actionValue << ", "
                << newActionValueEst << ") ";
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
    auto actionValues = strategy.estimatedActionValue();
    std::cout << std::setw(10) << name << "\t";
    for (auto const &v : actionValues) {
      std::cout << std::setw(10) << v << "\t";
    }
    std::cout << "\n";
  }

  return 0;
}