#pragma once
#include "exceptions.hpp"
#include <algorithm>
#include <cmath>
#include <iterator>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace strategy {

/** \defgroup Strategy Bases
 * @brief Strategy Bases are utilised in the creation of Strategy functors. They
 * contain the interface and data needed to keep track of the multi-armed-bandit
 * game.
 * @param actionValueEstimate Vector of action value estimates such that the
 * index corresponds to the action taken and the value is the estimate.
 * @param actionSelectionCount Vector of counts that each corresponding index
 * has had its' estimate updated.
 * @retval action to take
 */
/** @{ *you/

/** @brief Base class for action-value methods
 *
 */
template <typename TYPE_T,
          typename = typename std::enable_if<
              std::is_floating_point<TYPE_T>::value, TYPE_T>::type>
class ActionValueBase {

protected:
  std::vector<TYPE_T> actionValueEstimate;
  std::vector<std::size_t> actionSelectionCount;

public:
  ActionValueBase(std::vector<TYPE_T> actionValueEstimate,
                  std::vector<std::size_t> actionSelectionCount)
      : actionValueEstimate(actionValueEstimate),
        actionSelectionCount(actionSelectionCount){};
  template <typename> friend class PtrActionValueBase;
};

/** @brief Base class used to point to action value data. Derrived class
 * @details All aciton value functors inherit from this base class. Nullpointers
 * are used when the datum is not needed.
 */
template <typename TYPE_T,
          typename = typename std::enable_if<
              std::is_floating_point<TYPE_T>::value, TYPE_T>::type>
class PtrActionValueBase {
protected:
  std::vector<TYPE_T> *actionValueEstimate = nullptr;
  std::vector<std::size_t> *actionSelectionCount = nullptr;

public:
  PtrActionValueBase(){}; // Useful for methods that arent stateful
  PtrActionValueBase(std::vector<TYPE_T> *addrActionValueEstimate,
                     std::vector<std::size_t> *addrActionSelectionCount)
      : actionValueEstimate(actionValueEstimate),
        actionSelectionCount(actionSelectionCount) {
    checkEqualVectorDims(actionValueEstimate, actionSelectionCount);
  };
  PtrActionValueBase(std::vector<TYPE_T> &actionValueEstimate,
                     std::vector<std::size_t> &actionSelectionCount)
      : actionValueEstimate(&actionValueEstimate),
        actionSelectionCount(&actionSelectionCount) {
    checkEqualVectorDims(this->actionValueEstimate, this->actionSelectionCount);
  };
  /** @brief Only maintain pointer to the action value estimate */
  PtrActionValueBase(std::vector<TYPE_T> *addrActionValueEstimate)
      : actionValueEstimate(actionValueEstimate){};
  /** @brief Only maintain pointer to the action value estimate */
  PtrActionValueBase(std::vector<TYPE_T> &actionValueEstimate)
      : actionValueEstimate(&actionValueEstimate){};
  /** @brief Only maintain pointer to the value counts */
  PtrActionValueBase(std::vector<std::size_t> *addrActionSelectionCount)
      : actionSelectionCount(actionSelectionCount){};
  /** @brief Only maintain pointer to the value counts */
  PtrActionValueBase(std::vector<std::size_t> &actionSelectionCount)
      : actionSelectionCount(&actionSelectionCount){};
  PtrActionValueBase(ActionValueBase<TYPE_T> &actionValueBase)
      : actionValueEstimate(&(actionValueBase.actionValueEstimate)),
        actionSelectionCount(&(actionValueBase.actionSelectionCount)){};
};

/** @} */

} // namespace strategy

/** @brief Action Choices are how the strategic agent decides
 * whhether to explore of exploit.
 */
namespace strategy::action_choice {

enum class ActionTypes : int { EXPLORE, EXPLOIT };

template <typename TYPE_T,
          typename = typename std::enable_if<
              std::is_floating_point<TYPE_T>::value, TYPE_T>::type>
class ActionChoiceFunctor : PtrActionValueBase<TYPE_T> {

public:
  ActionChoiceFunctor() : PtrActionValueBase<TYPE_T>(){};
  ActionChoiceFunctor(std::vector<TYPE_T> &actionValueEstimate,
                      std::vector<std::size_t> &actionSelectionCount)
      : PtrActionValueBase<TYPE_T>(actionValueEstimate, actionSelectionCount){};
  virtual ActionTypes operator()() { return ActionTypes::EXPLOIT; };
};

/** @brief Action Choice Functor which selects a constant choice whenever
 * queried from operator().
 *
 * @details Some methodologies might only explore or exploit. In those cases we
 * can sepecify which methodology is being employed with this functor. For
 * example the "greedy" action value strategy always exploits.
 */
template <typename TYPE_T,
          typename = typename std::enable_if<
              std::is_floating_point<TYPE_T>::value, TYPE_T>::type>
class ConstantSelectionFunctor : public ActionChoiceFunctor<TYPE_T> {
private:
  ActionTypes action;

public:
  /**@param action Which action to select everytime the functor is queried */
  ConstantSelectionFunctor(const ActionTypes &action)
      : ActionChoiceFunctor<TYPE_T>(), action(action){};
  ActionTypes operator()() { return action; }
};

/** @brief Action choice Functor which selects whether to explore or exploit
 * according to a bernouli distributrion.
 *
 */
template <typename TYPE_T,
          typename = typename std::enable_if<
              std::is_floating_point<TYPE_T>::value, TYPE_T>::type>
class BinarySelectionFunctor : public ActionChoiceFunctor<TYPE_T> {
private:
  double prob;
  std::minstd_rand &generator;
  std::uniform_real_distribution<TYPE_T> distribution{0.0, 1.0};

public:
  BinarySelectionFunctor(const double &prob, std::minstd_rand &generator)
      : ActionChoiceFunctor<TYPE_T>(), prob(prob), generator(generator) {
    checkValidProbability(prob); // probabilities are between 0 and 1
  };
  ActionTypes operator()() override {
    return distribution(generator) > 1 - prob ? ActionTypes::EXPLORE
                                              : ActionTypes::EXPLOIT;
  }
};

} // namespace strategy::action_choice

/** @brief Methodology for choosing actions which aren't the immediate best */
namespace strategy::explore {

/** @brief Base functor for exploration methods */
template <typename TYPE_T,
          typename = typename std::enable_if<
              std::is_floating_point<TYPE_T>::value, TYPE_T>::type>
class ExploreFunctor {
protected:
  std::vector<TYPE_T> &actionValueEstimate;
  std::vector<std::size_t> &actionSelectionCount;
  std::size_t nActions;

public:
  ExploreFunctor(std::vector<TYPE_T> &actionValueEstimate,
                 std::vector<std::size_t> &actionSelectionCount)
      : actionValueEstimate(actionValueEstimate),
        actionSelectionCount(actionSelectionCount),
        nActions(actionSelectionCount.size()){};
  virtual std::size_t operator()() { return 0; };
};

/** @brief Select from nActions at random with equal probability. */
template <typename TYPE_T,
          typename = typename std::enable_if<
              std::is_floating_point<TYPE_T>::value, TYPE_T>::type>
class RandomActionSelectionFunctor : public ExploreFunctor<TYPE_T> {
private:
  std::uniform_int_distribution<std::size_t> distribution;
  std::minstd_rand &generator;

public:
  /** @brief Select from nActions at random with equal
   * probability.
   * @param nActions number of actions which can be chosen
   * from
   * @param generator random engine/generator
   */
  RandomActionSelectionFunctor(std::vector<TYPE_T> &actionValueEstimate,
                               std::vector<std::size_t> &actionSelectionCount,
                               std::minstd_rand &generator)
      : ExploreFunctor<TYPE_T>(actionValueEstimate, actionSelectionCount),
        distribution(std::uniform_int_distribution<std::size_t>(
            0, actionValueEstimate.size())),
        generator(generator){};
  /** @retval Random choice of action (index) */
  std::size_t operator()() { return distribution(generator); }
};

} // namespace strategy::explore

namespace strategy::step_size {

template <typename TYPE_T,
          typename = typename std::enable_if<
              std::is_floating_point<TYPE_T>::value, TYPE_T>::type>
class StepSizeFunctor {
protected:
  std::vector<std::size_t> &actionSelectionCount;

public:
  StepSizeFunctor(std::vector<std::size_t> &actionSelectionCount)
      : actionSelectionCount(actionSelectionCount){};
  // override for action
  virtual TYPE_T operator()(std::size_t action) { return 0; };
};

template <typename TYPE_T,
          typename = typename std::enable_if<
              std::is_floating_point<TYPE_T>::value, TYPE_T>::type>
class ConstantStepSizeFunctor : public StepSizeFunctor<TYPE_T> {
private:
  TYPE_T stepSize = 0;

public:
  ConstantStepSizeFunctor(std::vector<std::size_t> &actionSelectionCount,
                          TYPE_T stepSize)
      : strategy::step_size::StepSizeFunctor<TYPE_T>(actionSelectionCount),
        stepSize(stepSize){};
  TYPE_T operator()(std::size_t action) { return stepSize; }
};

template <typename TYPE_T,
          typename = typename std::enable_if<
              std::is_floating_point<TYPE_T>::value, TYPE_T>::type>
class SampleAverageStepSizeFunctor : public StepSizeFunctor<TYPE_T> {
public:
  SampleAverageStepSizeFunctor(std::vector<std::size_t> &actionSelectionCount)
      : StepSizeFunctor<TYPE_T>(actionSelectionCount){};
  TYPE_T operator()(std::size_t action) {
    return static_cast<TYPE_T>(1) /
           StepSizeFunctor<TYPE_T>::actionSelectionCount[action];
  }
};

} // namespace strategy::step_size

/** \defgroup actionSelectionFunctions
 *
 * @param actionValueEstimate Vector of action value estimates such that the
 * index corresponds to the action taken and the value is the estimate.
 * @retval action to take
 */
/** @{ */

/** @ingroup actionSelectionFunctions
 * @brief Methodology for exploiting curret knwoledge of actuion values.
 */
namespace strategy::exploit {

/** @ingroup actionSelectionFunctions
 * @brief Given a vector of real valued estimates for sample values return the
 * index corresponding to greatest estimate.
 * @details The greatest estimate is the so called greedy estimate. It is
 * considered the estimate which maximises utility given current knowledge of
 * the state of the value estimates. As such the selection is:
 *
 *  $$A_t = \text{argmax}_{a} ( Q_t(a) )$$
 *
 */
template <typename TYPE_T,
          typename = typename std::enable_if<
              std::is_floating_point<TYPE_T>::value, TYPE_T>::type>
std::size_t argmax(const std::vector<TYPE_T> &actionValueEstimate) {
  return std::distance(
      actionValueEstimate.begin(),
      std::max_element(actionValueEstimate.begin(), actionValueEstimate.end()));
}

/** @ingroup actionSelectionFunctions
 * @brief Given a vector of real values estimatess for a sample return
 * the index corresponding to the value with the greatest C confidence
 * interval
 * @details for the estimate given at each time step we only saw the
 * result of one action at a time. The selection is given by:
 *
 *  $$A_t = \text{argmax}_{a} \left\( Q_t(a) + c * \sqrt{ln(t) / N_t(a)}
 * \right)$$
 *
 * Where
 *  \f$t\f$ is the time step
 *  \f$A_t\f$ is the action selection at the current time step \f$t\f$
 *  \f$Q_t(a)\f$ is the value estimate for action a at time step \f$t\f$
 *  \f$c\f$ confidence interval quantile inverse (eg 1.96 for upper 95%)
 *  \f$N_t(a)\f$ is the number of times action a has been selected in \f$t\f$
 * time steps
 *
 * As \f$N_t(a)\f$ increases the variance of the estimate \f$Q_t(a)\f$
 * decreases. As \f$t\f$ increases (which includes the time steps for all the
 * times we didnt select a) the variance for the estamate \f$Q_t(a)\f$
 * increases. Hence we trade off reducing the estimate of a current action
 * against increasing the variance of all other estimates.
 *
 * @param confidenceQuantileInverse \f$c\f$: The value corresponding to a
 * specific quantile from a normal distribution for the confidence level we are
 * seeking. (eg 1.96 corresponds to 95% confidence)
 * @param actionValueEstimate \f$Q_t(a)\f$: Vector of action value estimates
 * such that the index corresponds to the action taken and the value is the
 * estimate.
 * @param actionSelectionCount \f$N_t(a)\f$: Vector of counts for how many times
 * value estimates have been updated using results for a specific action where
 * the index corresponds with action values.
 */
template <typename TYPE_T,
          typename = typename std::enable_if<
              std::is_floating_point<TYPE_T>::value, TYPE_T>::type>
std::size_t upperConfidenceBoundActionSelection(
    const TYPE_T &confidenceQuantileInverse,
    const std::vector<TYPE_T> &actionValueEstimate,
    const std::vector<std::size_t> &actionSelectionCount) {
  std::vector<TYPE_T> confidenceBound;
  confidenceBound.reserve(actionValueEstimate.size());
  const auto t = std::accumulate(actionSelectionCount.begin(),
                                 actionSelectionCount.end(), 0);
  for (int i = 0; i < actionValueEstimate.size(); i++) {
    confidenceBound.emplace_back(
        actionValueEstimate[i] +
        confidenceQuantileInverse *
            std::sqrt(std::log(t) / actionSelectionCount[i]));
  }
  return strategy::exploit::argmax(confidenceBound);
}

/** @} */

/** @brief Base Functor for action selection methods
 */
template <typename TYPE_T,
          typename = typename std::enable_if<

              std::is_floating_point<TYPE_T>::value, TYPE_T>::type>
class ExploitFunctor : public PtrActionValueBase<TYPE_T> {

public:
  ExploitFunctor() : PtrActionValueBase<TYPE_T>(){};
  ExploitFunctor(std::vector<TYPE_T> &actionValueEstimate)
      : PtrActionValueBase<TYPE_T>(actionValueEstimate){};
  ExploitFunctor(std::vector<TYPE_T> &actionValueEstimate,
                 std::vector<std::size_t> &actionSelectionCount)
      : PtrActionValueBase<TYPE_T>(actionValueEstimate, actionSelectionCount){};
  // Replace with exploit logic
  virtual std::size_t operator()() { return 0; };
};

template <typename TYPE_T,
          typename = typename std::enable_if<
              std::is_floating_point<TYPE_T>::value, TYPE_T>::type>
class ArgmaxFunctor : public ExploitFunctor<TYPE_T> {
public:
  ArgmaxFunctor(std::vector<TYPE_T> &actionValueEstimate)
      : ExploitFunctor<TYPE_T>(actionValueEstimate){};
  std::size_t operator()() override {
    return strategy::exploit::argmax(*(this->actionValueEstimate));
  }
};

template <typename TYPE_T,
          typename = typename std::enable_if<
              std::is_floating_point<TYPE_T>::value, TYPE_T>::type>
class UpperConfidenceBoundFunctor : public ExploitFunctor<TYPE_T> {
private:
  TYPE_T confidenceQuantileInverse;

public:
  UpperConfidenceBoundFunctor(std::vector<TYPE_T> &actionValueEstimate,
                              std::vector<std::size_t> &actionSelectionCount,
                              const TYPE_T &confidenceQuantileInverse)
      : ExploitFunctor<TYPE_T>(actionValueEstimate, actionSelectionCount),
        confidenceQuantileInverse(confidenceQuantileInverse){};
  std::size_t operator()() {
    return strategy::exploit::upperConfidenceBoundActionSelection(
        confidenceQuantileInverse, *(this->actionValueEstimate),
        *(this->actionSelectionCount));
  }
};

}; // namespace strategy::exploit

namespace strategy {

/** @ingroup actionSelectionFunctions
 * @brief Base interfaace with data attributes for a Strategy.
 * @details Contains definitions for the virtual interfaces for a strategy.
 */
template <typename TYPE_T,
          typename = typename std::enable_if<
              std::is_floating_point<TYPE_T>::value, TYPE_T>::type>
class StrategyBase {

public:
  StrategyBase(){};
  virtual std::size_t explore() = 0;
  virtual std::size_t exploit() = 0;
  /** @brief Explore or explot according to ther strategy.
   *  @retval The action index chosen either through exploration of exploitation
   */
  virtual std::size_t step() = 0;
  /** @brief Given a chosen action and returned Action value from the bandits
   * update the internal estimates for action values
   */
  virtual void update(std::size_t action, TYPE_T actionValue) = 0;
  virtual std::vector<TYPE_T> getActionValueEstimate() = 0;
  virtual TYPE_T getActionValueEstimate(std::size_t action) = 0;
  virtual std::vector<std::size_t> getActionSelectionCount() = 0;
  virtual std::size_t getActionSelectionCount(std::size_t action) = 0;
};

template <typename TYPE_T,
          class CHOICE = strategy::action_choice::ActionChoiceFunctor<TYPE_T>,
          class EXPLORE = strategy::explore::ExploreFunctor<TYPE_T>,
          class EXPLOIT = strategy::exploit::ExploitFunctor<TYPE_T>,
          class STEP = strategy::step_size::StepSizeFunctor<TYPE_T>,
          typename = typename std::enable_if<
              std::is_floating_point<TYPE_T>::value and
                  std::is_base_of<strategy::exploit::ExploitFunctor<TYPE_T>,
                                  EXPLOIT>::value,
              TYPE_T>::type>
class Strategy : public StrategyBase<TYPE_T> {

protected:
  std::vector<TYPE_T> actionValueEstimate;
  std::vector<std::size_t> actionSelectionCount;
  CHOICE choiceFunctor;
  EXPLORE exploreFunctor;
  EXPLOIT exploitFunctor;
  STEP stepSizeFunctor;

public:
  /** @brief Creating a strategy when the functors have been initialised
   * beforehand */
  Strategy(std::vector<TYPE_T> actionValueEstimate,
           std::vector<std::size_t> actionSelectionCount, CHOICE choiceFunctor,
           EXPLORE exploreFunctor, EXPLOIT exploitFunctor, STEP stepSizeFunctor)
      : actionValueEstimate(actionValueEstimate),
        actionSelectionCount(actionSelectionCount),
        choiceFunctor(choiceFunctor), exploreFunctor(exploreFunctor),
        exploitFunctor(exploitFunctor), stepSizeFunctor(stepSizeFunctor){};

  std::size_t explore() override { return this->exploreFunctor(); };
  std::size_t exploit() override { return this->exploitFunctor(); };
  /** @brief Explore or explot according to ther strategy.
   *  @retval The action index chosen either through exploration of exploitation
   */
  std::size_t step() override {
    return choiceFunctor() == strategy::action_choice::ActionTypes::EXPLORE
               ? exploreFunctor()
               : exploitFunctor();
  }
  /** @brief Given a chosen action and returned Action value from the bandits
   * update the internal estimates for action values
   */
  void update(std::size_t action, TYPE_T actionValue) override {
    actionValueEstimate[action] =
        (actionValueEstimate[action] +
         stepSizeFunctor(action) * (actionValue - actionValueEstimate[action]));
    ++actionSelectionCount[action];
  };
  std::vector<TYPE_T> getActionValueEstimate() override {
    return actionValueEstimate;
  }
  TYPE_T getActionValueEstimate(std::size_t action) override {
    return actionValueEstimate[action];
  }
  std::vector<std::size_t> getActionSelectionCount() override {
    return actionSelectionCount;
  }
  std::size_t getActionSelectionCount(std::size_t action) override {
    return actionSelectionCount[action];
  }
};

template <typename TYPE_T,
          typename = typename std::enable_if<
              std::is_floating_point<TYPE_T>::value, TYPE_T>::type>
class GreedyStrategy
    : public Strategy<
          TYPE_T, strategy::action_choice::ConstantSelectionFunctor<TYPE_T>,
          strategy::explore::ExploreFunctor<TYPE_T>,
          strategy::exploit::ArgmaxFunctor<TYPE_T>,
          strategy::step_size::SampleAverageStepSizeFunctor<TYPE_T>> {
public:
  GreedyStrategy(std::vector<TYPE_T> initialActionValueEstimate)
      : Strategy<TYPE_T,
                 strategy::action_choice::ConstantSelectionFunctor<TYPE_T>,
                 strategy::explore::ExploreFunctor<TYPE_T>,
                 strategy::exploit::ArgmaxFunctor<TYPE_T>,
                 strategy::step_size::SampleAverageStepSizeFunctor<TYPE_T>>(
            initialActionValueEstimate,
            std::vector<std::size_t>(initialActionValueEstimate.size(), 1),
            strategy::action_choice::ConstantSelectionFunctor<TYPE_T>(
                strategy::action_choice::ActionTypes::EXPLOIT),
            strategy::explore::ExploreFunctor<TYPE_T>(
                this->actionValueEstimate,
                this->actionSelectionCount), // NULL FUNCTOR
            strategy::exploit::ArgmaxFunctor<TYPE_T>(this->actionValueEstimate),
            strategy::step_size::SampleAverageStepSizeFunctor<TYPE_T>(
                this->actionSelectionCount)){};
};

template <typename TYPE_T,
          typename = typename std::enable_if<
              std::is_floating_point<TYPE_T>::value, TYPE_T>::type>
class EpsilonGreedyStrategy
    : public Strategy<
          TYPE_T, strategy::action_choice::BinarySelectionFunctor<TYPE_T>,
          strategy::explore::RandomActionSelectionFunctor<TYPE_T>,
          strategy::exploit::ArgmaxFunctor<TYPE_T>,
          strategy::step_size::SampleAverageStepSizeFunctor<TYPE_T>> {

public:
  EpsilonGreedyStrategy(std::minstd_rand &generator,
                        std::vector<TYPE_T> initialActionValueEstimate,
                        TYPE_T epsilon)
      : Strategy<TYPE_T,
                 strategy::action_choice::BinarySelectionFunctor<TYPE_T>,
                 strategy::explore::RandomActionSelectionFunctor<TYPE_T>,
                 strategy::exploit::ArgmaxFunctor<TYPE_T>,
                 strategy::step_size::SampleAverageStepSizeFunctor<TYPE_T>>(
            initialActionValueEstimate,
            std::vector<std::size_t>(initialActionValueEstimate.size(), 1),
            strategy::action_choice::BinarySelectionFunctor<TYPE_T>(epsilon,
                                                                    generator),
            strategy::explore::RandomActionSelectionFunctor<TYPE_T>(
                this->actionValueEstimate, this->actionSelectionCount,
                generator),
            strategy::exploit::ArgmaxFunctor<TYPE_T>(this->actionValueEstimate),
            strategy::step_size::SampleAverageStepSizeFunctor<TYPE_T>(
                this->actionSelectionCount)){};
};

} // namespace strategy