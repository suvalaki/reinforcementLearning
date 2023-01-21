#pragma once
#include "environment.hpp"
#include <cmath>
#include <exception>
#include <random>
#include <xtensor/xfixed.hpp>
#include <xtensor/xrandom.hpp>

#include "action.hpp"
#include "environment.hpp"
#include "greedy_policy.hpp"
#include "spec.hpp"
#include "state_action_value.hpp"

namespace policy {

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

  SETUP_TYPES(SINGLE_ARG(Policy<ENVIRON_T>));

  using KeyMaker = KEYMAPPER_T;
  using KeyType = typename KeyMaker::KeyType;
  using ValueType = VALUE_T;
  using StepSizeTaker = STEPSIZE_TAKER_T;

  using QTableValueType = ValueType;
  std::unordered_map<KeyType, QTableValueType, typename KeyMaker::Hash> q_table;

  typename ValueType::Factory valueFactory{};

  // Sample from the softmax distribution from the state to actions
  ActionSpace operator()(const StateType &s) override {

    if (q_table.empty()) {
      throw std::domain_error(
          "No actions have been taken yet - and the q table is "
          "not yet populated. Please initialise the q_table.");
    }

    auto norm = getSoftmaxNorm(s);
    auto r = xt::random::rand<double>({1})[0] * norm;
    auto it = q_table.begin();
    while (true) {
      r -= std::exp(it->second.value);
      if (r < 0) {
        break;
      }
      it++;
    }
    return KeyMaker::get_action_from_key(it->first);
  }

  virtual void update(const TransitionType &s) {
    auto reward = RewardType::reward(s);

    auto key = KeyMaker::make(s.state, s.action);
    if (q_table.find(key) != q_table.end()) {
      // Update the Q-table with the reward from the transition
      auto &v = q_table.at(key);
      // Sutton&Barto 2.8 Gradient Bandit Algorithms
      // H(t+1, A(t)) =
      // H(t, A(t)) + alpha * (R(t) - RAve(t)) * (1 - pi(t, A(t)))
      v.value = v.value + StepSizeTaker::getStepSize(v) *
                              (reward - valueFactory.averageReturn) *
                              (1 - getProbability(s.state, key));
      v.step++;
    } else {
      q_table.emplace(key, valueFactory.create(0.0F, 1));
    }

    // update all other actions
    for (auto &[k, v] : q_table) {
      if (k != key) {
        v.value = v.value + StepSizeTaker::getStepSize(v) *
                                (reward - valueFactory.averageReturn) *
                                (-getProbability(s.state, k));
      }
    }

    valueFactory.update(reward);
  }

  /// @brief Norm over the potential reachable actions from this state
  PrecisionType getSoftmaxNorm(const StateType &s) const {
    return std::accumulate(q_table.begin(), q_table.end(), 0.0F,
                           [](const auto &v, const auto &p) {
                             return v + std::exp(p.second.value);
                           });
  }
  PrecisionType getSoftmaxNorm() const {
    return std::accumulate(q_table.begin(), q_table.end(), 0.0F,
                           [](const auto &v, const auto &p) {
                             return v + std::exp(p.second.value);
                           });
  }
  PrecisionType getProbability(const StateType &s, const KeyType &key) const {
    return std::exp(q_table.at(key).value) / getSoftmaxNorm(s);
  }
  PrecisionType getProbability(const KeyType &key) const {
    return std::exp(q_table.at(key).value) / getSoftmaxNorm();
  }

  void printQTable() const {

    std::cout << "QTable\n=====\n";
    for (const auto &[k, v] : q_table) {

      std::cout << k << "\t" << v.value << "\t" << v.step << "\t"
                << getProbability(k) << "\n";
    }
  }
};

} // namespace policy
