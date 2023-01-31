#pragma once
#include "environment.hpp"
#include <cmath>
#include <exception>
#include <limits>
#include <random>
#include <xtensor/xfixed.hpp>
#include <xtensor/xrandom.hpp>

#include "action.hpp"
#include "environment.hpp"
#include "greedy_policy.hpp"
#include "random_policy.hpp"
#include "spec.hpp"
#include "state_action_value.hpp"

namespace policy {

constexpr auto min_policy_value = -10.0F;
constexpr auto max_policy_value = 10.0F;

template <environment::EnvironmentType ENVIRON_T>
struct DistributionStateActionValue : public StateActionValue<ENVIRON_T> {

  SETUP_TYPES(StateActionValue<ENVIRON_T>)
  using EnvironmentType = ENVIRON_T;

  using BaseType::BaseType;

  struct Factory {
    using ValueType = DistributionStateActionValue;
    PrecisionType averageReturn = 0;
    std::size_t step = 0;
    DistributionStateActionValue create(const PrecisionType &value = 0,
                                        const std::size_t &step = 0) {
      return DistributionStateActionValue{value, step};
    }

    void update(const PrecisionType &reward) {
      averageReturn = (averageReturn * step + reward) / (step + 1);
      step++;
    }

    void update() {}
  };
};

template <typename T>
concept isDistributionStateActionValue =
    (std::is_base_of_v<
         DistributionStateActionValue<typename T::EnvironmentType>, T> ||
     std::is_same_v<DistributionStateActionValue<typename T::EnvironmentType>,
                    T>)&&isStateActionValueFactory<typename T::Factory>;

template <environment::EnvironmentType ENVIRON_T,
          isStateActionKeymaker KEYMAPPER_T = DefaultActionKeymaker<ENVIRON_T>,
          isDistributionStateActionValue VALUE_T =
              DistributionStateActionValue<ENVIRON_T>,
          step_size_taker STEPSIZE_TAKER_T =
              weighted_average_step_size_taker<VALUE_T>>
struct DistributionPolicy
    : GreedyPolicy<ENVIRON_T, KEYMAPPER_T, VALUE_T, STEPSIZE_TAKER_T> {

  SETUP_KEYPOLICY_TYPES(SINGLE_ARG(
      GreedyPolicy<ENVIRON_T, KEYMAPPER_T, VALUE_T, STEPSIZE_TAKER_T>));

  void initialise(EnvironmentType &environ, const std::size_t &iterations) {
    auto randomPolicy = policy::RandomPolicy<ENVIRON_T>();

    for (int i = 0; i < iterations; i++) {
      auto recommendedAction = randomPolicy(environ.state);
      auto transition = environ.step(recommendedAction);
      // update this policy with the result of the random init
      update(environ, transition);
      environ.update(transition);
    }

    environ.reset();
  }

  // Sample from the softmax distribution from the state to actions
  ActionSpace operator()(const EnvironmentType &e, const StateType &s) {

    if (this->empty()) {
      throw std::domain_error(
          "No actions have been taken yet - and the q table is "
          "not yet populated. Please initialise the this->");
    }

    auto norm = getSoftmaxNorm(e, s);
    auto r = xt::random::rand<double>({1})[0] * norm;
    auto it = this->begin();
    while (true) {
      r -= std::exp(it->second.value);
      if (r < 0) {
        break;
      }
      it++;
    }
    return KeyMaker::get_action_from_key(it->first);
  }

  virtual void update(const EnvironmentType &e, const TransitionType &s) {
    auto reward = RewardType::reward(s);

    auto key = KeyMaker::make(s.state, s.action);
    if (this->find(key) != this->end()) {
      // Update the Q-table with the reward from the transition
      auto &v = this->at(key);
      // Sutton&Barto 2.8 Gradient Bandit Algorithms
      // H(t+1, A(t)) =
      // H(t, A(t)) + alpha * (R(t) - RAve(t)) * (1 - pi(t, A(t)))
      v.value = v.value + StepSizeTaker::getStepSize(v) *
                              (reward - this->valueFactory.averageReturn) *
                              (1 - getProbability(e, s.state, key));
      v.step++;
    } else {
      this->emplace(key, this->valueFactory.create(0.0F, 1));
    }

    // update all other actions
    for (auto &[k, v] : *this) {
      if (k != key) {
        v.value = v.value + StepSizeTaker::getStepSize(v) *
                                (reward - this->valueFactory.averageReturn) *
                                (-getProbability(e, s.state, k));
      }
    }

    this->valueFactory.update(reward);
  }

  /// @brief Norm over the potential reachable actions from this state
  PrecisionType getSoftmaxNorm(const EnvironmentType &e,
                               const StateType &s) const {
    auto reachableActions = e.getReachableActions(s);
    auto norm =
        std::accumulate(reachableActions.begin(), reachableActions.end(), 0.0F,
                        [this, &s](const auto &v, const auto &a) {
                          auto key = KeyMaker::make(s, a);
                          if (this->find(key) == this->end()) {
                            return v;
                          }
                          return v + std::exp(this->at(key).value);
                        });
    return norm;
  }
  PrecisionType getProbability(const EnvironmentType &e, const StateType &s,
                               const KeyType &key) const {
    return std::exp(this->at(key).value) / getSoftmaxNorm(e, s);
  }

  std::vector<std::pair<KeyType, PrecisionType>>
  getProbabilities(const EnvironmentType &e, const StateType &s) const {
    std::vector<std::pair<KeyType, PrecisionType>> probs;
    for (const auto &[k, v] : *this) {
      probs.emplace_back(k, getProbability(e, s, k));
    }
    return probs;
  }

  void setProbability(const EnvironmentType &e, const StateType &s,
                      const KeyType &key, const PrecisionType &p) {

    this->at(key).value = std::log(p * getSoftmaxNorm(e, s));
  }

  void setDeterministicPolicy(const EnvironmentType &e, const StateType &s,
                              const KeyType &key) {

    auto reachaleActions = e.getReachableActions(s);

    for (const auto &a : reachaleActions) {
      auto k = KeyMaker::make(s, a);

      if (this->find(k) == this->end()) {
        continue;
      }

      if (k == key) {
        this->at(k).value = max_policy_value;
      } else {
        this->at(k).value = min_policy_value;
      }
    }
  }

  void printQTable(const EnvironmentType &e) const {

    std::cout << "QTable\n=====\n";
    for (const auto &[k, v] : *this) {

      auto s = KeyMaker::get_state_from_key(e, k);

      std::cout << k << "\t" << v.value << "\t" << v.step << "\t"
                << getProbability(e, s, k) << "\n";
    }
  }
};

template <typename T>
concept isDistributionPolicy = std::is_base_of_v<
    DistributionPolicy<typename T::EnvironmentType, typename T::KeyMaker,
                       typename T::ValueType, typename T::StepSizeTaker>,
    T>;

} // namespace policy
