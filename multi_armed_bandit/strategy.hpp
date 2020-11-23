#include "bandit.hpp"
#include <limits>
#include <numeric>

template <typename Iter, typename Function>
Iter argmax(Iter begin, Iter end, Function f) {
  typedef typename std::iterator_traits<Iter>::value_type T;
  return std::min_element(begin, end,
                          [&f](const T &a, const T &b) { return f(a) < f(b); });
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
private:
  NormalPriorNormalMultiArmedBandit<TYPE_T> &bandits;
  std::size_t time_step;

public:
  MultiArmedBanditStrategy(NormalPriorNormalMultiArmedBandit<TYPE_T> &bandits)
      : bandits(bandits){};
  virtual std::size_t actionChosenCount(std::size_t action);
  // vector of all actionChosen
  virtual std::vector<std::size_t> actionChosenCount();
  virtual TYPE_T estimatedActionValue(std::size_t action);
  // vector of all estimatedActionValues
  virtual std::vector<TYPE_T> estimatedActionValue();
  virtual std::size_t explore();
  virtual std::size_t exploit();
  virtual std::size_t getAction();
  virtual void update(TYPE_T banditValue, std::size_t action);
};

template <typename TYPE_T>
class GreedyStrategy : public ActionValueStrategy<TYPE_T> {

private:
  /** Each action has a reward value which we can estimate and define.
   *  The greedy method will end up picking that with the highest
   *   estimated action value.
   *
   *      Q_t ()
   */
  std::size_t bestIdx;
  std::vector<double> estActionValue;
  std::vector<std::size_t> actionCnts;

public:
  GreedyStrategy(NormalPriorNormalMultiArmedBandit<TYPE_T> &bandits,
                 std::vector<TYPE_T> estActionValue)
      : ActionValueStrategy<Type_T>(bandits), estActionValue(estActionValue) {
    estActionValue.resize(bandits.getNBandits);
    actionCnts.resize(bandits.getNBandits);
    bestIdx = argmax(estActionValue.begin(), estActionValue.end(),
                     [](TYPE_T x) { return x; });
  };
  std::size_t actionChosenCount(std::size_t action) {
    return actionCnts[action];
  };
  std::vector<std::size_t> actionChosenCount() { return actionCnts; };
  TYPE_T estimatedActionValue(std::size_t action) {
    return estActionValue[action];
  };
  std::vector<TYPE_T> estimatedActionValue() { return estActionValue; };
  std::size_t explore() { return bestIdx; };
  std::size_t exploit() { return bestIdx; };
  std::size_t getAction() { return exploit(); };
  void update(TYPE_T banditValue, std::size_t action){/* do nothing */};
};

template <typename TYPE_T>
class EpsilonGreedyStrategy : public strategies::GreedyStrategy<Type_T> {

private:
  std::minstd_rand &generator;
  double epsilon;
  std::uniform_real_distribution<double> exploreExploitDist(0.0, 1.0);
  std::uniform_int_distribution<std::size_t> exploreChoiceDist;

public:
  EpsilonGreedyStrategy(NormalPriorNormalMultiArmedBandit<TYPE_T> &bandits,
                        std::minstd_rand &generator,
                        std::vector<TYPE_T> initialActionValueEstimate,
                        double epsilon)
      : strategies::ActionValueStrategy(bandits), generator(generator),
        exploreChoiceDist(std::uniform_int_distribution<std::size_t>(
            0, bandits.getNBandits())),
        estActionValue(initialActionValueEstimate), epsilon(epsilon){};
  std::size_t exploit() {
    return argmax(estActionValue.begin(), estActionValue.end(),
                  [](TYPE_T x) { return x; });
  }
  std::size_t explore() {
    // pick an index at random
    return exploreChoiceDist(generator);
  }
  std::size_t getAction() {
    return (exploreExploitDist(generator) < 1 - epsilon) ? exploit()
                                                         : explore();
  }
  void update(TYPE_T banditValue, std::size_t action) {
    // update the action value estimates
    // Q(a, n+1) = sum_(n+1) [v(a, i) / (n+1)]
    //           = sum_(n) [v(a, i) / (n+1)] + v(a, n+1) / (n+1)
    //           = n * sum_(n) [v(a, i) / n] / (n+1) + v(a, n+1) / (n+1)
    //           = n * Q(a, n) / (n + 1) + v(a, n+1) / (n+1)
    //           = [n * Q(a, n) + v(a, n+1)] / (n+1)
    estActionValue[action] =
        (actionCnts[action] * estActionValue[action] + banditValue) /
        (actionCnts[action] + 1);
    ++actionCnts[action];
    ++time_step;
  };
};

} // namespace strategies