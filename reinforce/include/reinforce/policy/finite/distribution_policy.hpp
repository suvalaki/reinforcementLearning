#pragma once

#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "reinforce/markov_decision_process/finite_transition_model.hpp"
#include "reinforce/policy/distribution_policy.hpp"
#include "reinforce/policy/finite/policy.hpp"
#include "reinforce/policy/finite/value_policy.hpp"
#include "reinforce/policy/objectives/finite_value.hpp"
#include "reinforce/policy/objectives/step_size.hpp"
#include "reinforce/policy/value.hpp"

#define FDP FiniteDistributionPolicy<VALUE_FUNCTION_T>

namespace policy {

template <objectives::isFiniteValueFunction VALUE_FUNCTION_T>
struct FiniteDistributionPolicy : virtual DistributionPolicy<typename VALUE_FUNCTION_T::EnvironmentType>,
                                  FinitePolicyValueFunctionMixin<VALUE_FUNCTION_T>

{
  using BaseType = DistributionPolicy<typename VALUE_FUNCTION_T::EnvironmentType>;
  SETUP_TYPES_W_VALUE_FUNCTION(VALUE_FUNCTION_T);
  using ValueFunctionBaseType = typename ValueFunctionType::ValueFunctionBaseType;
  using StepSizeTaker = typename ValueFunctionType::StepSizeTaker;

  // Pull in the direct functions we need since they have different signatures
  using ValueFunctionType::operator();
  using DistributionPolicy<typename ValueFunctionType::EnvironmentType>::operator();

  static constexpr auto min_policy_value = -10.0F;
  static constexpr auto max_policy_value = 10.0F;

  FiniteDistributionPolicy(auto &&...args) : FinitePolicyValueFunctionMixin<VALUE_FUNCTION_T>(args...) {}
  FiniteDistributionPolicy(const FiniteDistributionPolicy &p) : FinitePolicyValueFunctionMixin<VALUE_FUNCTION_T>(p) {}

  void update(const EnvironmentType &e, const TransitionType &s) override;

  PrecisionType getProbability(const EnvironmentType &e, const StateType &s, const ActionSpace &a) const override;
  PrecisionType getLogProbability(const EnvironmentType &e, const StateType &s, const ActionSpace &a) const override;
  PrecisionType getKernel(const EnvironmentType &e, const StateType &s, const ActionSpace &a) const override;
  PrecisionType getNormalisationConstant(const EnvironmentType &e, const StateType &s) const override;
  ActionSpace sampleAction(const EnvironmentType &e, const StateType &s) const override;

  /// @brief Norm over the potential reachable actions from this state
  PrecisionType getSoftmaxNorm(const EnvironmentType &e, const StateType &s) const;
  ActionSpace getArgmaxAction(const EnvironmentType &e, const StateType &s) const override;

  std::enable_if_t<
      environment::MarkovDecisionEnvironmentType<EnvironmentType>,
      std::vector<std::pair<KeyType, PrecisionType>>>
  getProbabilities(const EnvironmentType &e, const StateType &s) const;

  void setDeterministicPolicy(const EnvironmentType &e, const StateType &s, const ActionSpace &a);
};

template <objectives::isFiniteValueFunction VALUE_FUNCTION_T>
void FDP::update(const EnvironmentType &e, const TransitionType &s) {}

template <objectives::isFiniteValueFunction VALUE_FUNCTION_T>
typename FDP::PrecisionType
FDP::getProbability(const EnvironmentType &e, const StateType &s, const ActionSpace &a) const {
  const auto key = KeyMaker::make(e, s, a);
  if (this->find(key) == this->end())
    return 0.0F;
  return std::exp(this->at(key).value) / getSoftmaxNorm(e, s);
}

template <objectives::isFiniteValueFunction VALUE_FUNCTION_T>
typename FDP::PrecisionType
FDP::getLogProbability(const EnvironmentType &e, const StateType &s, const ActionSpace &a) const {
  const auto key = KeyMaker::make(e, s, a);
  if (this->find(key) == this->end())
    return -std::numeric_limits<PrecisionType>::infinity();
  return this->at(key).value - std::log(getSoftmaxNorm(e, s));
}

template <objectives::isFiniteValueFunction VALUE_FUNCTION_T>
typename FDP::PrecisionType FDP::getKernel(const EnvironmentType &e, const StateType &s, const ActionSpace &a) const {
  const auto key = KeyMaker::make(e, s, a);
  return this->at(key).value;
}

template <objectives::isFiniteValueFunction VALUE_FUNCTION_T>
typename FDP::PrecisionType FDP::getNormalisationConstant(const EnvironmentType &e, const StateType &s) const {
  return getSoftmaxNorm(e, s);
}

template <objectives::isFiniteValueFunction VALUE_FUNCTION_T>
typename FDP::ActionSpace FDP::sampleAction(const EnvironmentType &e, const StateType &s) const {
  return this->getArgmaxAction(e, s);
}

template <objectives::isFiniteValueFunction VALUE_FUNCTION_T>
typename FDP::PrecisionType FDP::getSoftmaxNorm(const EnvironmentType &e, const StateType &s) const {

  // get the softmax norm
  auto reachableActions = e.getReachableActions(s);
  auto norm = std::accumulate(
      reachableActions.begin(), reachableActions.end(), 0.0F, [this, &e, &s](const auto &v, const auto &a) {
        auto key = KeyMaker::make(e, s, a);
        if (this->find(key) == this->end()) {
          return v;
        }
        return v + std::exp(this->at(key).value);
      });
  return norm;
}

template <objectives::isFiniteValueFunction VALUE_FUNCTION_T>
typename FDP::ActionSpace FDP::getArgmaxAction(const EnvironmentType &e, const StateType &s) const {
  return FinitePolicyValueFunctionMixin<VALUE_FUNCTION_T>::getArgmaxAction(e, s);
};

template <objectives::isFiniteValueFunction VALUE_FUNCTION_T>
std::enable_if_t<
    environment::MarkovDecisionEnvironmentType<typename FDP::EnvironmentType>,
    std::vector<std::pair<typename FDP::KeyType, typename FDP::PrecisionType>>>
FDP::getProbabilities(const EnvironmentType &e, const StateType &s) const {
  std::vector<std::pair<KeyType, PrecisionType>> probs;
  for (const auto &[k, v] : *this) {
    if (KeyMaker::get_state_from_key(e, k) != s) {
      continue;
    }
    probs.emplace_back(k, getProbability(e, s, KeyMaker::get_action_from_key(e, k)));
  }
  return probs;
}

template <objectives::isFiniteValueFunction VALUE_FUNCTION_T>
void FDP::setDeterministicPolicy(const EnvironmentType &e, const StateType &s, const ActionSpace &a0) {

  auto reachaleActions = e.getReachableActions(s);

  const auto key = KeyMaker::make(e, s, a0);
  for (const auto &a : reachaleActions) {
    auto k = KeyMaker::make(e, s, a);

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

template <
    environment::FiniteEnvironmentType E,
    template <typename> typename KEYMAKER_C = objectives::StateActionKeymaker,
    template <typename> typename VALUE_C = objectives::FiniteValue,
    template <typename> typename INCREMENTAL_STEPSIZE_C = objectives::weighted_average_step_size_taker,
    auto INITIAL_VALUE = 0.0F,
    auto DISCOUNT_RATE = 0.0F>
using FiniteDistributionPolicyC = FiniteDistributionPolicy<objectives::FiniteValueFunction<
    objectives::ValueFunction<KEYMAKER_C<E>, VALUE_C<E>, INITIAL_VALUE, DISCOUNT_RATE>,
    INCREMENTAL_STEPSIZE_C<VALUE_C<E>>>>;

} // namespace policy

#undef FDP