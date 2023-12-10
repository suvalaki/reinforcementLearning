#pragma once
#include <cmath>
#include <exception>

#include "environment.hpp"
#include "policy/random_policy.hpp"

#define FRP FiniteRandomPolicy<E>

namespace policy {

// The finite random policy differs from the generic random policy because it has additional information
// about the size of state and action spaces. This allows it to generate random events over the bounded
// specification. When more information is known about the transition model of the environment (for example)
// the number of actions available to be taken from a given state the normalisation on the kernel can be
// calculated. This allows the policy to be used in a more efficient in sampling and evaluation.
// Furthermore, this additional information allows us to propogate additional information into downstream
// tasks - for example mote carlo importance sampling.

template <environment::FiniteEnvironmentType E>
struct FiniteRandomPolicy : RandomPolicy<E> {

  SETUP_TYPES_FROM_ENVIRON(SINGLE_ARG(E));

  // Get a random event over the bounded specification
  virtual void update(const EnvironmentType &e, const TransitionType &s){};

  PrecisionType getProbability(const EnvironmentType &e, const StateType &s, const ActionSpace &a) const override;
  PrecisionType getLogProbability(const EnvironmentType &e, const StateType &s, const ActionSpace &a) const override;
  PrecisionType getKernel(const EnvironmentType &e, const StateType &s, const ActionSpace &a) const override;
  PrecisionType getNormalisationConstant(const EnvironmentType &e, const StateType &s) const override;
  ActionSpace getArgmaxAction(const EnvironmentType &e, const StateType &s) const override;
};

template <environment::FiniteEnvironmentType E>
typename FRP::PrecisionType
FRP::getProbability(const EnvironmentType &e, const StateType &s, const ActionSpace &a) const {
  return this->getKernel(e, s, a) / this->getNormalisationConstant(e, s);
}

template <environment::FiniteEnvironmentType E>
typename FRP::PrecisionType
FRP::getLogProbability(const EnvironmentType &e, const StateType &s, const ActionSpace &a) const {
  return -std::log(this->getNormalisationConstant(e, s));
}

template <environment::FiniteEnvironmentType E>
typename FRP::PrecisionType FRP::getKernel(const EnvironmentType &e, const StateType &s, const ActionSpace &a) const {
  return 1.0;
}

template <environment::FiniteEnvironmentType E>
typename FRP::PrecisionType FRP::getNormalisationConstant(const EnvironmentType &e, const StateType &s) const {
  return e.getReachableActions(s).size();
}

template <environment::FiniteEnvironmentType E>
typename FRP::ActionSpace FRP::getArgmaxAction(const EnvironmentType &e, const StateType &s) const {
  throw std::logic_error("A purely random policy has no notion of a 'best' action.");
}

} // namespace policy

#undef FRP