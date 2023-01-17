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

template <typename T>
concept PolicyType = std::is_base_of_v<Policy<typename T::EnvironmentType>, T>;

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

template <typename T> struct HashBuilder {
  std::size_t operator()(const typename T::KeyType &key) const {
    return T::hash(key);
  }
};

// use crtp
template <environment::EnvironmentType ENVIRON_T, typename KEY_T>
struct StateActionKeymaker {
  using EnvironmentType = typename ENVIRON_T::EnvironmentType;
  using StateType = typename EnvironmentType::StateType;
  using ActionSpace = typename EnvironmentType::ActionSpace;

  // The type that will be used for Q value lookup
  using KeyType = KEY_T;

  static KeyType make(const StateType &s, const ActionSpace &action);
  static ActionSpace get_action_from_key(const KeyType &key);
  static std::size_t hash(const KEY_T &key);

  using Hash = HashBuilder<StateActionKeymaker<ENVIRON_T, KEY_T>>;
};

template <typename T>
concept isStateActionKeymaker = requires(T t) {
  typename T::KeyType;
  {
    T::make(std::declval<typename T::StateType>(),
            std::declval<typename T::ActionSpace>())
    } -> std::same_as<typename T::KeyType>;
  {
    T::get_action_from_key(std::declval<typename T::KeyType>())
    } -> std::same_as<typename T::ActionSpace>;
  { T::hash(std::declval<typename T::KeyType>()) } -> std::same_as<std::size_t>;
};

template <environment::EnvironmentType ENVIRON_T>
struct DefaultActionKeymaker
    : StateActionKeymaker<ENVIRON_T,
                          std::pair<typename ENVIRON_T::StateType,
                                    typename ENVIRON_T::ActionSpace>> {

  using baseType =
      StateActionKeymaker<ENVIRON_T,
                          std::pair<typename ENVIRON_T::StateType,
                                    typename ENVIRON_T::ActionSpace>>;

  using EnvironmentType = typename baseType::EnvironmentType;
  using StateType = typename baseType::StateType;
  using ActionSpace = typename baseType::ActionSpace;
  using KeyType = typename baseType::KeyType;

  static KeyType make(const StateType &s, const ActionSpace &action) {
    return std::make_pair(s, action);
  }

  static typename ENVIRON_T::ActionSpace
  get_action_from_key(const std::pair<typename ENVIRON_T::StateType,
                                      typename ENVIRON_T::ActionSpace> &key) {
    return key.second;
  }

  static std::size_t
  hash(const std::pair<typename ENVIRON_T::StateType,
                       typename ENVIRON_T::ActionSpace> &key) {
    return key.first.hash() ^ key.second.hash();
  }

  using Hash = HashBuilder<DefaultActionKeymaker<ENVIRON_T>>;
};

// Helper to print the default key type
template <typename T0, typename T1>
std::ostream &operator<<(std::ostream &os, const std::pair<T0, T1> &p) {
  os << "(" << p.first << ", " << p.second << ")";
  return os;
}

template <environment::EnvironmentType ENVIRON_T,
          isStateActionKeymaker KEYMAPPER_T = DefaultActionKeymaker<ENVIRON_T>>
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

  using QTableValueType =
      std::tuple<typename EnvironmentType::RewardType::PrecisionType, int>;
  std::unordered_map<KeyType, QTableValueType, typename KeyMaker::Hash> q_table;

  // Search over a space of actions and return the one with the highest
  // reward
  ActionSpace operator()(const StateType &s) override {
    PrecisionType maxVal = 0;
    auto action = random_spec_gen<
        typename ActionSpace::SpecType>(); // start with a random action so we
                                           // at least have one that is
                                           // permissible

    for (auto &[k, v] : q_table) {
      if (maxVal < std::get<0>(v)) {
        maxVal = std::get<0>(v);
        action = KeyMaker::get_action_from_key(k);
      }
    }

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
      std::get<0>(v) =
          (std::get<0>(v) * std::get<1>(v) + reward) / (std::get<1>(v) + 1);
      std::get<1>(v)++;
    } else {
      q_table.emplace(key, QTableValueType{reward, 1});
    }
  };

  virtual PrecisionType greedyValue() {
    PrecisionType maxVal = 0;
    for (auto &[k, v] : q_table) {
      if (maxVal < std::get<0>(v)) {
        maxVal = std::get<0>(v);
      }
    }
    return maxVal;
  }

  void printQTable() const {

    std::cout << "QTable\n=====\n";
    for (const auto &[k, v] : q_table) {

      std::cout << k << "\t" << std::get<0>(v) << "\t" << std::get<1>(v)
                << "\n";
    }
  }
};

template <environment::EnvironmentType ENVIRON_T,
          PolicyType EXPLOIT_POLICY =
              GreedyPolicy<ENVIRON_T, DefaultActionKeymaker<ENVIRON_T>>,
          class E = xt::random::default_engine_type>
struct EpsilonGreedyPolicy : EXPLOIT_POLICY {
  using baseType = EXPLOIT_POLICY;
  using EnvironmentType = typename baseType::EnvironmentType;
  using StateType = typename baseType::StateType;
  using ActionSpace = typename baseType::ActionSpace;
  using TransitionType = typename baseType::TransitionType;
  using RewardType = typename EnvironmentType::RewardType;
  using PrecisionType = typename RewardType::PrecisionType;

  RandomPolicy<ENVIRON_T> randomPolicy;

  PrecisionType epsilon = 0.1;
  E &engine;

  EpsilonGreedyPolicy(PrecisionType epsilon = 0.1,
                      E &engine = xt::random::get_default_random_engine())
      : epsilon(epsilon), engine(engine) {}

  ActionSpace explore(const StateType &s) { return randomPolicy(s); }

  ActionSpace exploit(const StateType &s) {
    return static_cast<EXPLOIT_POLICY &>(*this)(s);
  }

  ActionSpace operator()(const StateType &s) override {
    if (xt::random::rand<double>(xt::xshape<1>{}, 0, 1, engine)[0] < epsilon) {
      return explore(s);
    } else {
      return exploit(s);
    }
  }
};

} // namespace policy
