#pragma once
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "environment.hpp"

namespace environment {

// Finite markov decision processes rely on a KNOWN  transition model
// from state,action -> next state
// As such the model has limited usefulness

template <StepType STEP_T, RewardType REWARD_T, ReturnType RETURN_T>
struct MarkovDecisionEnvironment : Environment<STEP_T, REWARD_T, RETURN_T> {

  SETUP_TYPES(SINGLE_ARG(Environment<STEP_T, REWARD_T, RETURN_T>));

  using TransitionModel = std::unordered_map<TransitionType, PrecisionType,
                                             typename TransitionType::Hash>;
  /// @brief  The mapping from (state, action, nextState) to probabiliies
  TransitionModel transitionModel;

  // For sampling
  std::random_device rd;
  std::mt19937 gen{rd()};

  MarkovDecisionEnvironment() = delete;
  MarkovDecisionEnvironment(const TransitionModel &t) : transitionModel(t){};
  MarkovDecisionEnvironment(const TransitionModel &t, const StateType s)
      : BaseType(s), transitionModel(t){};

  TransitionType step(const ActionSpace &action) override {

    // sample next state according to the transition model
    auto nextStates = getReachableStates(this->state, action);
    auto nextStatesProbabilities =
        getTransitionProbabilities(this->state, action);
    auto nextState = sample(nextStates, nextStatesProbabilities);

    return TransitionType{this->state, action, nextState};
  }

  std::vector<TransitionType> getTransitions(const StateType &s,
                                             const ActionSpace &a) const {
    std::vector<TransitionType> transitions;
    for (const auto &t : transitionModel) {
      if (t.first.state == s and t.first.action == a) {
        transitions.push_back(t.first);
      }
    }
    return transitions;
  }

  std::vector<ActionSpace> getReachableActions(const StateType &s) const {
    std::vector<ActionSpace> actions;
    for (const auto &t : transitionModel) {
      if (t.first.state == s) {
        actions.push_back(t.first.action);
      }
    }
    return actions;
  }

  std::vector<StateType> getReachableStates(const StateType &s,
                                            const ActionSpace &a) const {
    std::vector<StateType> states;
    for (const auto &t : transitionModel) {
      if (t.first.state == s and t.first.action == a) {
        states.push_back(t.first.nextState);
      }
    }
    return states;
  }

  std::vector<PrecisionType> getTransitionProbabilities(const StateType &s,
                                                        const ActionSpace &a) {
    std::vector<PrecisionType> probabilities;
    for (const auto &t : transitionModel) {
      if (t.first.state == s and t.first.action == a) {
        probabilities.push_back(t.second);
      }
    }
    return probabilities;
  }

  StateType sample(const std::vector<StateType> &states,
                   const std::vector<PrecisionType> &probabilities) {
    std::discrete_distribution<size_t> distribution(probabilities.begin(),
                                                    probabilities.end());
    return states[distribution(gen)];
  }

  /// @brief Get all possible states under the finite transation model
  std::unordered_set<StateType, typename StateType::Hash>
  getAllPossibleStates() const {
    std::unordered_set<StateType, typename StateType::Hash> states;
    for (const auto &t : transitionModel) {
      states.emplace(t.first.state);
    }
    return states;
  }
};

template <typename ENVIRON_T>
concept MarkovDecisionEnvironmentType =
    std::is_base_of_v<MarkovDecisionEnvironment<typename ENVIRON_T::StepType,
                                                typename ENVIRON_T::RewardType,
                                                typename ENVIRON_T::ReturnType>,
                      ENVIRON_T>;

} // namespace environment
