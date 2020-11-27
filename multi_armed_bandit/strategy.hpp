#pragma once
#include "bandit.hpp"
#include <algorithm>
#include <limits>
#include <numeric>

template <typename TYPE_T>
std::size_t argmax(std::vector<TYPE_T> &sampleArray) {
  return std::distance(
      sampleArray.begin(),
      std::max_element(sampleArray.begin(), sampleArray.end()));
}

namespace value_estimate {

template <typename TYPE_T>
TYPE_T sampleAverage(std::vector<TYPE_T> actionValues,
                     std::size_t actionCount) {

  return actionCount > 0
             ? std::accumulate(actionValues.begin(), actionValues.end(), 0) /
                   actionCount
             : std::numeric_limits<TYPE_T>::min();
}

template <typename TYPE_T>
TYPE_T sampleAverage(TYPE_T actionValueSum, std::size_t actionCount) {
  return actionCount > 0 ? actionValueSum / actionCount
                         : std::numeric_limits<TYPE_T>::min();
}

} // namespace value_estimate

namespace strategies {

enum class Strategies : int {
  GREEDY,
  EPSILON_GREEDY,
  EPSILON_GREEDY_NON_STATIONAIRY,
  UPPER_CONFIDENCE_BOUNDS
};

/** Baseclass for action value strategies over the multi-armed-bandit.
 *  Action value strategies:
 *
 *   - let r_t(a) be the reward for taking action a at time t (stationairy or
 *     not)
 *   - let q(a) be the true value for taking action a. The true value is the
 *     expected reward
 *   - let Q_t(a) be the estimate for q(a) at time step t
 *   - let N_t(a) be the number of time periods that the actionValueMethodf has
 *     chosen action a after t time periods.
 *
 */
template <typename TYPE_T> class ActionValueStrategy {
protected:
  NormalPriorNormalMultiArmedBandit<TYPE_T> &bandits;
  std::size_t time_step = 0;
  std::vector<TYPE_T> estActionValue;
  std::vector<std::size_t> actionCnts;

public:
  ActionValueStrategy(NormalPriorNormalMultiArmedBandit<TYPE_T> &bandits,
                      std::vector<TYPE_T> &estActionValue)
      : bandits(bandits), estActionValue(estActionValue) {
    this->actionCnts.resize(bandits.getNBandits());
    std::fill(actionCnts.begin(), actionCnts.end(), 1);
  };
  virtual std::size_t actionChosenCount(std::size_t action) = 0;
  // vector of all actionChosen
  virtual std::vector<std::size_t> actionChosenCount() = 0;
  virtual TYPE_T estimatedActionValue(std::size_t action) = 0;
  // vector of all estimatedActionValues
  virtual std::vector<TYPE_T> estimatedActionValue() = 0;
  virtual std::size_t explore() = 0;
  virtual std::size_t exploit() = 0;
  virtual std::size_t getAction() = 0;
  void update(TYPE_T banditValue, std::size_t action) {
    // update the action value estimates
    // Q(a, n+1) = sum_(n+1) [v(a, i) / (n+1)]
    //           = sum_(n) [v(a, i) / (n+1)] + v(a, n+1) / (n+1)
    //           = n * sum_(n) [v(a, i) / n] / (n+1) + v(a, n+1) / (n+1)
    //           = n * Q(a, n) / (n + 1) + v(a, n+1) / (n+1)
    //           = [n * Q(a, n) + v(a, n+1)] / (n+1)
    this->estActionValue[action] =
        ((this->actionCnts[action] * this->estActionValue[action] +
          banditValue) /
         (this->actionCnts[action] + 1));
    ++this->actionCnts[action];
    ++this->time_step;
  };
};

template <typename TYPE_T>
class GreedyStrategy : public ActionValueStrategy<TYPE_T> {

  /** Each action has a reward value which we can estimate and define.
   *  The greedy method will end up picking that with the highest
   *   estimated action value.
   *
   *      Q_t ()
   */

public:
  GreedyStrategy(NormalPriorNormalMultiArmedBandit<TYPE_T> &bandits,
                 std::vector<TYPE_T> &estActionValue)
      : ActionValueStrategy<TYPE_T>(bandits, estActionValue){};
  std::size_t actionChosenCount(std::size_t action) override {
    return this->actionCnts[action];
  };
  std::vector<std::size_t> actionChosenCount() override {
    return this->actionCnts;
  };
  TYPE_T estimatedActionValue(std::size_t action) override {
    return this->estActionValue[action];
  };
  std::vector<TYPE_T> estimatedActionValue() override {
    return this->estActionValue;
  };
  std::size_t explore() override {
    return argmax<TYPE_T>(this->estActionValue);
  };
  std::size_t exploit() override {
    return argmax<TYPE_T>(this->estActionValue);
  };
  std::size_t getAction() override { return exploit(); };
  void update(TYPE_T banditValue, std::size_t action) {
    ActionValueStrategy<TYPE_T>::update(banditValue, action);
  }
};

template <typename TYPE_T>
class EpsilonGreedyStrategy : public strategies::GreedyStrategy<TYPE_T> {

private:
  std::minstd_rand &generator;
  double epsilon;
  std::uniform_real_distribution<double> exploreExploitDist{0.0, 1.0};
  std::uniform_int_distribution<std::size_t> exploreChoiceDist;

public:
  EpsilonGreedyStrategy(NormalPriorNormalMultiArmedBandit<TYPE_T> &bandits,
                        std::minstd_rand &generator,
                        std::vector<TYPE_T> initialActionValueEstimate,
                        double epsilon)
      : strategies::GreedyStrategy<TYPE_T>(bandits, initialActionValueEstimate),
        generator(generator),
        exploreChoiceDist(std::uniform_int_distribution<std::size_t>(
            0, bandits.getNBandits())),
        epsilon(epsilon){};
  std::size_t exploit() { return argmax(this->estActionValue); }
  std::size_t explore() {
    // pick an index at random
    return exploreChoiceDist(generator);
  }
  std::size_t getAction() {
    return (exploreExploitDist(generator) < 1.0L - epsilon) ? exploit()
                                                            : explore();
  }
};

} // namespace strategies