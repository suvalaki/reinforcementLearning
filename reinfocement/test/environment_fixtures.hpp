#pragma once

#include "environment.hpp"
#include "markov_decision_process/finite_transition_model.hpp"

namespace fixtures {

template <std::size_t N, std::size_t M> struct simple_environment_builder {

  using StateType0 = state::State<float, spec::CompositeArraySpec<spec::BoundedAarraySpec<int, 0, N, 1>>>;
  using ActionType0 = action::Action<StateType0, spec::CompositeArraySpec<spec::BoundedAarraySpec<int, 0, M, 1>>>;
  using StepType0 = step::Step<ActionType0>;
  using RewardType0 = reward::Reward<ActionType0>;
  using ReturnType0 = returns::Return<RewardType0>;

  struct type : environment::FiniteEnvironment<StepType0, RewardType0, ReturnType0> {
    SETUP_TYPES(SINGLE_ARG(environment::FiniteEnvironment<StepType0, RewardType0, ReturnType0>));
    StateType reset() override { return StateType{}; }
    StateType stateFromIndex(std::size_t i) const override { return StateType{i, {}}; }
    ActionSpace actionFromIndex(std::size_t i) const override { return ActionSpace{i}; }
    std::unordered_set<StateType, typename StateType::Hash> getAllPossibleStates() const override {
      std::unordered_set<StateType, typename StateType::Hash> states;
      for (std::size_t i = 0; i < N; ++i) {
        states.insert(StateType{i, {}});
      }
      return states;
    };
    std::unordered_set<ActionSpace, typename ActionSpace::Hash> getAllPossibleActions() const override {
      std::unordered_set<ActionSpace, typename ActionSpace::Hash> actions;
      for (std::size_t i = 0; i < M; ++i) {
        actions.insert(ActionSpace{i});
      }
      return actions;
    };
    StateType getNullState() const override { return StateType{0, {}}; }
  };
};

template <std::size_t N, std::size_t M>
using simple_environment_builder_t = typename simple_environment_builder<N, M>::type;

template <std::size_t N, std::size_t M> struct simple_markov_environment_builder {

  using StateType0 = state::State<float, spec::CompositeArraySpec<spec::BoundedAarraySpec<int, 0, N, 1>>>;
  using ActionType0 = action::Action<StateType0, spec::CompositeArraySpec<spec::BoundedAarraySpec<int, 0, M, 1>>>;
  using StepType0 = step::Step<ActionType0>;
  using RewardType0 = reward::Reward<ActionType0>;
  using ReturnType0 = returns::Return<RewardType0>;

  struct type : environment::MarkovDecisionEnvironment<StepType0, RewardType0, ReturnType0> {
    SETUP_TYPES(SINGLE_ARG(environment::MarkovDecisionEnvironment<StepType0, RewardType0, ReturnType0>));
    using typename BaseType::TransitionModel;

    constexpr static typename TransitionModel::TransitionModelMap makeTransitionModel() {
      // We assume a model were from every state you can reach withn weighted prob from any action.
      // But the number of actions that we can use is limited by the state index. ActionIndex <= FromStateIndex
      typename TransitionModel::TransitionModelMap tm;
      for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < M; ++j) {
          for (std::size_t k = 0; k < N; ++k) {
            if (k > i or j > i) //
              continue;
            tm.emplace(TransitionType{StateType(i, {}), ActionSpace(j), StateType(k, {})},
                       // Weighted distribtion by action index j / (i * sum j=0...M)
                       static_cast<float>(j) / static_cast<float>(i * (M * (M - 1) / 2)));
          }
        }
      }
      return tm;
    }

    constexpr static std::array<StateType, N> makeStateSpace() {
      std::array<StateType, N> states;
      for (std::size_t i = 0; i < N; ++i) {
        states[i] = StateType{i, {}};
      }
      return states;
    }

    constexpr static std::array<ActionSpace, M> makeActionSpace() {
      std::array<ActionSpace, M> actions;
      for (std::size_t i = 0; i < M; ++i) {
        actions[i] = ActionSpace{i};
      }
      return actions;
    }

    type() : BaseType(TransitionModel{makeTransitionModel(), makeStateSpace(), makeActionSpace()}) {} // constructor

    StateType reset() override { return StateType{}; }
    StateType stateFromIndex(std::size_t i) const override { return StateType{i, {}}; }
    ActionSpace actionFromIndex(std::size_t i) const override { return ActionSpace{i}; }
    std::unordered_set<StateType, typename StateType::Hash> getAllPossibleStates() const override {
      std::unordered_set<StateType, typename StateType::Hash> states;
      for (std::size_t i = 0; i < N; ++i) {
        states.insert(StateType{i, {}});
      }
      return states;
    };
    std::unordered_set<ActionSpace, typename ActionSpace::Hash> getAllPossibleActions() const override {
      std::unordered_set<ActionSpace, typename ActionSpace::Hash> actions;
      for (std::size_t i = 0; i < M; ++i) {
        actions.insert(ActionSpace{i});
      }
      return actions;
    };
    StateType getNullState() const override { return StateType{0, {}}; }
  };
};

template <std::size_t N, std::size_t M>
using simple_markov_environment_builder_t = typename simple_markov_environment_builder<N, M>::type;

using S1A1 = simple_environment_builder_t<1, 1>;
using S1A2 = simple_environment_builder_t<1, 2>;
using S2A2 = simple_environment_builder_t<2, 2>;

using MS5A2 = simple_markov_environment_builder_t<5, 2>;
using MS5A10 = simple_markov_environment_builder_t<5, 10>;

} // namespace fixtures
