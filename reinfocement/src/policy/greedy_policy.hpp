#pragma once
#include <algorithm>
#include <random>
#include <xtensor/xfixed.hpp>
#include <xtensor/xrandom.hpp>

#include "action.hpp"
#include "environment.hpp"
#include "policy.hpp"
#include "random_policy.hpp"
#include "spec.hpp"

namespace policy {

using spec::CompositeArraySpecType;
using spec::isBoundedArraySpec;
using spec::isCategoricalArraySpec;

template <environment::EnvironmentType ENVIRON_T,
          isStateActionKeymaker KEYMAPPER_T = DefaultActionKeymaker<ENVIRON_T>,
          isStateActionValue VALUE_T = StateActionValue<ENVIRON_T>,
          step_size_taker STEPSIZE_TAKER_T =
              weighted_average_step_size_taker<VALUE_T>>
struct GreedyPolicy : Policy<ENVIRON_T> {
  using baseType = Policy<ENVIRON_T>;
  using EnvironmentType = typename baseType::EnvironmentType;
  using StateType = typename baseType::StateType;
  using ActionSpace = typename baseType::ActionSpace;
  using TransitionType = typename baseType::TransitionType;
  using RewardType = typename EnvironmentType::RewardType;
  using PrecisionType = typename RewardType::PrecisionType;

  using KeyMaker = KEYMAPPER_T;
  using KeyType = typename KeyMaker::KeyType;
  using ValueType = VALUE_T;
  using StepSizeTaker = STEPSIZE_TAKER_T;

  using QTableValueType = ValueType;
  std::unordered_map<KeyType, QTableValueType, typename KeyMaker::Hash> q_table;

  typename ValueType::Factory valueFactory{};

  // Search over a space of actions and return the one with the highest
  // reward
  ActionSpace operator()(const StateType &s) override {
    auto maxVal = valueFactory.create();
    auto action = random_spec_gen<
        typename ActionSpace::SpecType>(); // start with a random action so we
                                           // at least have one that is
                                           // permissible

    if (q_table.empty()) {
      return action;
    }

    auto maxIdx = std::max_element(
        q_table.begin(), q_table.end(),
        [](const auto &p1, const auto &p2) { return p1.second < p2.second; });

    action = KeyMaker::get_action_from_key(maxIdx->first);

    return action;
  }

  // Update the Q-table with the new transition
  virtual void update(const TransitionType &s) {

    // Reward for this transition
    auto reward = RewardType::reward(s);

    auto key = KeyMaker::make(s.state, s.action);
    if (q_table.find(key) != q_table.end()) {
      // Update the Q-table with the reward from the transition
      auto &v = q_table.at(key);
      // Replace with the updated monte carlo avergae
      v.value = v.value + StepSizeTaker::getStepSize(v) * (reward - v.value);
      v.step++;
    } else {
      q_table.emplace(key, valueFactory.create(reward, 1));
    }

    // Go over all the other actions and update them with their global callback
    // A better mechanism might be to start with a state factory that holds
    // global state
    valueFactory.update();
  };

  virtual PrecisionType greedyValue() {
    PrecisionType maxVal = 0;
    for (auto &[k, v] : q_table) {
      if (maxVal < v.value) {
        maxVal = v.value;
      }
    }
    return maxVal;
  }

  void printQTable() const {

    std::cout << "QTable\n=====\n";
    for (const auto &[k, v] : q_table) {

      std::cout << k << "\t" << v.value << "\t" << v.step << "\n";
    }
  }
};

} // namespace policy