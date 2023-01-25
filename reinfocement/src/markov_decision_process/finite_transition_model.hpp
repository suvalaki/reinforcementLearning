#pragma once
#include <unordered_map>

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

  MarkovDecisionEnvironment() = delete;
  MarkovDecisionEnvironment(const TransitionModel &t) : transitionModel(t){};
  MarkovDecisionEnvironment(const TransitionModel &t, const StateType s)
      : BaseType(s), transitionModel(t){};
};

template <typename ENVIRON_T>
concept MarkovDecisionEnvironmentType =
    std::is_base_of_v<MarkovDecisionEnvironment<typename ENVIRON_T::StepType,
                                                typename ENVIRON_T::RewardType,
                                                typename ENVIRON_T::ReturnType>,
                      ENVIRON_T>;

} // namespace environment
