#pragma once
#include <array>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "environment.hpp"

namespace environment {

// Finite markov decision processes rely on a KNOWN  transition model
// from state,action -> next state
// As such the model has limited usefulness

template <StepType STEP_T, RewardType REWARD_T, ReturnType RETURN_T, std::size_t N_STATES, std::size_t N_ACTIONS>
struct MarkovDecisionEnvironment : FiniteEnvironment<STEP_T, REWARD_T, RETURN_T, N_STATES, N_ACTIONS> {

  SETUP_TYPES(SINGLE_ARG(FiniteEnvironment<STEP_T, REWARD_T, RETURN_T, N_STATES, N_ACTIONS>));
  using EnvironmentType = MarkovDecisionEnvironment;

  using BaseType::nActions;
  using BaseType::nStates;

  // TODO : Make the unordered map constexpr...?
  struct TransitionModel {
    using TransitionModelMap = std::unordered_map<TransitionType, PrecisionType, typename TransitionType::Hash>;
    TransitionModelMap transitions;
    std::array<StateType, N_STATES> states;
    std::array<ActionSpace, N_ACTIONS> actions;
  };

  /// @brief  The mapping from (state, action, nextState) to probabiliies
  TransitionModel transitionModel;

  // TODO : CONVERT TRANSITION MODEL INTO TRANSITION MATRIX - to enable INDEXING VIA NUMBER
  // BY NECESSITY the transition model already has to know everything about state-action pairs. Because it
  // uses the transitionType hash.

  // For sampling
  std::random_device rd;
  std::mt19937 gen{rd()};

  MarkovDecisionEnvironment() = delete;
  MarkovDecisionEnvironment(const TransitionModel &t) : transitionModel(t){};
  MarkovDecisionEnvironment(const TransitionModel &t, const StateType &s) : BaseType(s), transitionModel(t){};

  StateType stateFromIndex(std::size_t idx) const override { return transitionModel.states[idx]; };
  ActionSpace actionFromIndex(std::size_t idx) const override { return transitionModel.actions[idx]; };

  TransitionType step(const ActionSpace &action) override {

    // sample next state according to the transition model
    auto nextStates = getReachableStates(this->state, action);
    auto nextStatesProbabilities = getTransitionProbabilities(this->state, action);
    auto nextState = sample(nextStates, nextStatesProbabilities);

    return TransitionType{this->state, action, nextState};
  }

  std::vector<TransitionType> getTransitions(const StateType &s, const ActionSpace &a) const {
    std::vector<TransitionType> transitions;
    for (const auto &t : transitionModel.transitions) {
      if (t.first.state == s and t.first.action == a) {
        transitions.push_back(t.first);
      }
    }
    return transitions;
  }

  std::unordered_set<ActionSpace, typename ActionSpace::Hash> getReachableActions(const StateType &s) const {
    std::unordered_set<ActionSpace, typename ActionSpace::Hash> actions;
    for (const auto &t : transitionModel.transitions) {
      actions.emplace(t.first.action);
    }
    return actions;
  }

  std::unordered_set<StateType, typename StateType::Hash> getReachableStates(const StateType &s,
                                                                             const ActionSpace &a) const {
    std::unordered_set<StateType, typename StateType::Hash> states;
    for (const auto &t : transitionModel.transitions) {
      if (t.first.state == s and t.first.action == a) {
        states.emplace(t.first.nextState);
      }
    }
    return states;
  }

  std::vector<PrecisionType> getTransitionProbabilities(const StateType &s, const ActionSpace &a) {
    std::vector<PrecisionType> probabilities;
    for (const auto &t : transitionModel.transitions) {
      if (t.first.state == s and t.first.action == a) {
        probabilities.push_back(t.second);
      }
    }
    return probabilities;
  }

  StateType sample(const std::unordered_set<StateType, typename StateType::Hash> &states,
                   const std::vector<PrecisionType> &probabilities) {
    std::discrete_distribution<size_t> distribution(probabilities.begin(), probabilities.end());
    return *std::next(states.begin(), distribution(gen));
  }

  /// @brief Get all possible states under the finite transation model
  std::unordered_set<StateType, typename StateType::Hash> getAllPossibleStates() const {
    std::unordered_set<StateType, typename StateType::Hash> states;
    for (const auto &t : transitionModel.transitions) {
      states.emplace(t.first.state);
    }
    return states;
  }

  std::unordered_set<ActionSpace, typename ActionSpace::Hash> getAllPossibleActions() const {
    std::unordered_set<ActionSpace, typename ActionSpace::Hash> actions;
    for (const auto &t : transitionModel.transitions) {
      actions.emplace(t.first.action);
    }
    return actions;
  }
};

template <typename ENVIRON_T>
concept MarkovDecisionEnvironmentType = std::is_base_of_v<MarkovDecisionEnvironment<typename ENVIRON_T::StepType,
                                                                                    typename ENVIRON_T::RewardType,
                                                                                    typename ENVIRON_T::ReturnType,
                                                                                    ENVIRON_T::nStates,
                                                                                    ENVIRON_T::nActions>,
                                                          ENVIRON_T>;

} // namespace environment
