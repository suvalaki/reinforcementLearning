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
  double eps = 0.1;
  strategies::EpsilonGreedyStrategy<double> strategy(
      bandits, generator, initialValueEstimates, eps);

  for (size_t i = 0; i < n_samples; i++) {
    if (i == 0) {
      std::cout << "Generatimng Ranom Numbers"
                << " with " << n_bandits << " bandits and " << n_samples
                << " samples\n";
    } else {
      std::cout << "\n";
    }
    std::vector<double> result = bandits.sample(generator);
    std::size_t action = strategy.getAction();
    double actionValue = result[action];
    strategy.update(actionValue, action);
    double newActionValueEst = strategy.estimatedActionValue(action);
    for (auto const &v : result) {
      std::cout << std::setw(9) << v << " ";
    }
    std::cout << "Action: " << action << " valueEstimate:" << newActionValueEst;
  }
  std::cout << "\n\nFinal Action Value Estimates";
  std::cout << "\n";
  auto actionValues = strategy.estimatedActionValue();
  for (auto const &v : actionValues) {
    std::cout << std::setw(9) << v << " ";
  }
  std::cout << "\n";

  return 0;
}