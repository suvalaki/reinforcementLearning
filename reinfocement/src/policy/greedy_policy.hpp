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

#include "policy/value.hpp"

namespace policy {

using spec::CompositeArraySpecType;
using spec::isBoundedArraySpec;
using spec::isCategoricalArraySpec;

#define SETUP_POLICY_TYPES(BASE_T, KEYMAPPER_T, VALUE_T)                                                               \
  SETUP_TYPES(SINGLE_ARG(BASE_T))                                                                                      \
  using EnvironmentType = typename BaseType::EnvironmentType;                                                          \
  using KeyMaker = KEYMAPPER_T;                                                                                        \
  using KeyType = typename KeyMaker::KeyType;                                                                          \
  using ValueType = VALUE_T;

template <environment::EnvironmentType ENVIRON_T,
          isStateActionKeymaker KEYMAPPER_T = DefaultActionKeymaker<ENVIRON_T>,
          isStateActionValue VALUE_T = StateActionValue<ENVIRON_T>,
          step_size_taker STEPSIZE_TAKER_T = weighted_average_step_size_taker<VALUE_T>>
struct GreedyPolicy : Policy<ENVIRON_T>,
                      FiniteValueFunctionMixin<ValueFunctionPrototype<ENVIRON_T, KEYMAPPER_T, 0.0F, 0.0F>, VALUE_T> {

  SETUP_POLICY_TYPES(SINGLE_ARG(Policy<ENVIRON_T>), SINGLE_ARG(KEYMAPPER_T), SINGLE_ARG(VALUE_T));
  using StepSizeTaker = STEPSIZE_TAKER_T;
  using ValueFunctionType =
      FiniteValueFunctionMixin<ValueFunctionPrototype<ENVIRON_T, KEYMAPPER_T, 0.0F, 0.0F>, VALUE_T>;

  typename ValueType::Factory valueFactory{};

  // Search over a space of actions and return the one with the highest
  // reward
  ActionSpace operator()(const StateType &s) override {
    auto maxVal = valueFactory.create();
    auto action = random_spec_gen<typename ActionSpace::SpecType>(); // start with a random action so we
                                                                     // at least have one that is
                                                                     // permissible

    if (this->empty()) {
      return action;
    }

    auto maxIdx = std::max_element(
        this->begin(), this->end(), [](const auto &p1, const auto &p2) { return p1.second < p2.second; });

    action = KeyMaker::get_action_from_key(maxIdx->first);

    return action;
  }

  ActionSpace operator()(const EnvironmentType &e, const StateType &s) { return (*this)(s); }

  // Update the Q-table with the new transition
  virtual void update(const TransitionType &s) {

    // Reward for this transition
    auto reward = RewardType::reward(s);

    auto key = KeyMaker::make(s.state, s.action);
    if (this->find(key) != this->end()) {
      // Update the Q-table with the reward from the transition
      auto &v = this->at(key);
      // Replace with the updated monte carlo avergae
      v.value = v.value + StepSizeTaker::getStepSize(v) * (reward - v.value);
      v.step++;
    } else {
      this->emplace(key, valueFactory.create(reward, 1));
    }

    // Go over all the other actions and update them with their global callback
    // A better mechanism might be to start with a state factory that holds
    // global state
    valueFactory.update();
  };

  virtual PrecisionType greedyValue() {
    PrecisionType maxVal = 0;
    for (auto &[k, v] : *this) {
      if (maxVal < v.value) {
        maxVal = v.value;
      }
    }
    return maxVal;
  }

  void printQTable() const {

    std::cout << "QTable\n=====\n";
    for (const auto &[k, v] : *this) {

      std::cout << k << "\t" << v.value << "\t" << v.step << "\n";
    }
  }
};

#define SETUP_KEYPOLICY_TYPES(POLICY_T)                                                                                \
  SETUP_TYPES(SINGLE_ARG(POLICY_T))                                                                                    \
  using EnvironmentType = typename BaseType::EnvironmentType;                                                          \
  using KeyMaker = typename BaseType::KeyMaker;                                                                        \
  using KeyType = typename KeyMaker::KeyType;                                                                          \
  using ValueType = typename BaseType::ValueType;                                                                      \
  using StepSizeTaker = typename BaseType::StepSizeTaker;

template <typename T>
concept isGreedyPolicy = std::is_base_of_v<
    GreedyPolicy<typename T::EnvironmentType, typename T::KeyMaker, typename T::ValueType, typename T::StepSizeTaker>,
    T>;

} // namespace policy