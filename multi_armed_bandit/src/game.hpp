#ifndef MAB_GAME_H
#define MAB_GAME_H

#include "bandit.hpp"
#include "strategy.hpp"

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace multi_armed_bandit {

template <typename TYPE_T,
          typename = typename std::enable_if<
              std::is_floating_point<TYPE_T>::value, TYPE_T>::type>
class Game {
private:
  std::minstd_rand generator;
  MultiArmedBandit<TYPE_T> bandits;
  std::map<std::string, strategy::StrategyBase<TYPE_T>> strategies;

public:
  Game(std::size_t n_bandits, std::size_t);

  void addStrategy(const std::string &name,
                   strategy::StrategyBase<TYPE_T> strategy) {
    strategies.emplace(name, std::move(strategy));
  }
};

} // namespace multi_armed_bandit

#endif