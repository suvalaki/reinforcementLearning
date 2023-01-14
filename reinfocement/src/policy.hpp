#pragma once
#include "action.hpp"
#include "environment.hpp"
#include <random>
#include <xtensor/xfixed.hpp>
#include <xtensor/xrandom.hpp>

#include "action.hpp"
#include "spec.hpp"

namespace policy {

using spec::CompositeArraySpecType;
using spec::isBoundedArraySpec;
using spec::isCategoricalArraySpec;

template <environment::EnvironmentType ENVIRON_T> struct Policy {
  using EnvironmentType = ENVIRON_T;
  using PrecisionType = typename EnvironmentType::PrecisionType;
  using StateType = typename EnvironmentType::StateType;
  using ActionSpace = typename EnvironmentType::ActionSpace;
  using TransitionType = typename EnvironmentType::TransitionType;

  // Run the policy over the current state of the environment
  virtual ActionSpace operator()(const StateType &s) = 0;
  virtual void update(const TransitionType &s) = 0;
};

template <isBoundedArraySpec T, class E = xt::random::default_engine_type>
std::enable_if_t<isBoundedArraySpec<T>, typename T::DataType>
random_spec_gen(E &engine = xt::random::get_default_random_engine()) {

  if constexpr (std::is_integral_v<typename T::ValueType>)
    return xt::random::randint(T::shape, T::min, T::max, engine);

  else if constexpr (std::is_floating_point_v<typename T::ValueType>)
    return xt::random::rand(T::shape, T::min, T::max, engine);

  else
    return xt::zeros(T::shape);
};

template <isCategoricalArraySpec T, class E = xt::random::default_engine_type>
std::enable_if_t<isCategoricalArraySpec<T>, typename T::DataType>
random_spec_gen(E &engine = xt::random::get_default_random_engine()) {
  return xt::random::randint(T::shape, T::min, T::max, engine);
};

template <CompositeArraySpecType T, class E = xt::random::default_engine_type>
requires CompositeArraySpecType<T> std::enable_if_t < CompositeArraySpecType<T>,
typename T::DataType >
    random_spec_gen(E &engine = xt::random::get_default_random_engine()) {

  // Tuple of the random types
  return [&engine]<std::size_t... N>(std::index_sequence<N...>) {
    return typename T::DataType(
        random_spec_gen<std::tuple_element_t<N, typename T::tupleType>>(
            engine)...);
  }
  (std::make_index_sequence<std::tuple_size_v<typename T::tupleType>>());
}

template <environment::EnvironmentType ENVIRON_T>
struct RandomPolicy : Policy<ENVIRON_T> {
  using baseType = Policy<ENVIRON_T>;
  using EnvironmentType = typename baseType::EnvironmentType;
  using StateType = typename baseType::StateType;
  using ActionSpace = typename baseType::ActionSpace;
  using TransitionType = typename baseType::TransitionType;

  // Get a random event over the bounded specification
  ActionSpace operator()(const StateType &s) override {
    return ActionSpace{random_spec_gen<typename ActionSpace::SpecType>()};
  }

  virtual void update(const TransitionType &s){};
};

struct pair_hash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2> &pair) const {
    return pair.first.hash() ^ pair.second.hash();
  }
};

template <environment::EnvironmentType ENVIRON_T>
struct GreedyPolicy : Policy<ENVIRON_T> {
  using baseType = Policy<ENVIRON_T>;
  using EnvironmentType = typename baseType::EnvironmentType;
  using StateType = typename baseType::StateType;
  using ActionSpace = typename baseType::ActionSpace;
  using TransitionType = typename baseType::TransitionType;
  using RewardType = typename EnvironmentType::RewardType;
  using PrecisionType = typename RewardType::PrecisionType;

  using QTableValueType =
      std::tuple<StateType, ActionSpace,
                 typename EnvironmentType::RewardType::PrecisionType, int>;
  std::unordered_map<std::pair<StateType, ActionSpace>, QTableValueType,
                     pair_hash>
      q_table;

  // Search over a space of actions and return the one with the highest
  // reward
  ActionSpace operator()(const StateType &s) override {
    PrecisionType maxVal = 0;
    auto action = random_spec_gen<
        typename ActionSpace::SpecType>(); // start with a random action so we
                                           // at least have one that is
                                           // permissible

    for (auto &[k, v] : q_table) {
      if (maxVal < std::get<2>(v)) {
        maxVal = std::get<2>(v);
        action = std::get<1>(v);
      }
    }

    return action;
  }

  // Update the Q-table with the new transition
  virtual void update(const TransitionType &s) {

    // Reward for this transition
    auto reward = RewardType::reward(s);

    auto key = std::make_pair(s.state, s.action);
    if (q_table.find(key) != q_table.end()) {
      // Update the Q-table with the reward from the transition
      auto &v = q_table.at(key);
      // Replace with the updated monte carlo avergae
      std::get<2>(v) = (std::get<2>(v) * std::get<3>(v) +  reward) / (std::get<3>(v) + 1) ;
      std::get<3>(v)++;
    } else {
      q_table.emplace(key, QTableValueType{s.state, s.action, reward, 1});
    }
  };

  virtual PrecisionType greedyValue() {
    PrecisionType maxVal = 0;
    for (auto &[k, v] : q_table) {
      if (maxVal < std::get<2>(v)) {
        maxVal = std::get<2>(v);
      }
    }
    return maxVal;
  }

  void printQTable() const {

    std::cout << "QTable\n=====\n";
    for (const auto& [k,v] : q_table){

      std::cout << std::get<0>(v) << "\t" << std::get<1>(v) << "\t" << 
      std::get<2>(v) << "\t" << std::get<3>(v) << "\n" ;

    }

  }

};

} // namespace policy
