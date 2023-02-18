#pragma once
#include <algorithm>
#include <limits>
#include <random>
#include <xtensor/xfixed.hpp>
#include <xtensor/xrandom.hpp>

#include "environment.hpp"
#include "policy/policy.hpp"
#include "policy/value.hpp"

#define GDT GreedyDistributionMixin<E>
#define GPT GreedyPolicy<KEYMAPPER_T, VALUE_T, INITIAL_VALUE, DISCOUNT_RATE>

namespace policy {

template <environment::EnvironmentType E> struct GreedyDistributionMixin : virtual PolicyDistributionMixin<E> {

  using BaseType = PolicyDistributionMixin<E>;
  SETUP_TYPES_FROM_NESTED_ENVIRON(SINGLE_ARG(BaseType::EnvironmentType));

  PrecisionType getProbability(const EnvironmentType &e, const StateType &s, const ActionSpace &a) const override;
  PrecisionType getLogProbability(const EnvironmentType &e, const StateType &s, const ActionSpace &a) const override;
  PrecisionType getKernel(const EnvironmentType &e, const StateType &s, const ActionSpace &a) const override;
  PrecisionType getNormalisationConstant(const EnvironmentType &e, const StateType &s) const override;
  ActionSpace sampleAction(const EnvironmentType &e, const StateType &s) const override;
};

template <environment::EnvironmentType E>
typename GDT::PrecisionType
GreedyDistributionMixin<E>::getProbability(const EnvironmentType &e, const StateType &s, const ActionSpace &a) const {
  if (a == this->getArgmaxAction(e, s)) {
    return 1.0F;
  } else {
    return 0.0F;
  }
}

template <environment::EnvironmentType E>
typename GDT::PrecisionType GreedyDistributionMixin<E>::getLogProbability(const EnvironmentType &e,
                                                                          const StateType &s,
                                                                          const ActionSpace &a) const {
  if (a == this->getArgmaxAction(e, s)) {
    return 0.0F;
  } else {
    return -std::numeric_limits<PrecisionType>::infinity();
  }
}

template <environment::EnvironmentType E>
typename GDT::PrecisionType
GreedyDistributionMixin<E>::getKernel(const EnvironmentType &e, const StateType &s, const ActionSpace &a) const {
  if (a == this->getArgmaxAction(e, s)) {
    return 1.0F;
  } else {
    return 0.0F;
  }
}

template <environment::EnvironmentType E>
typename GDT::PrecisionType GreedyDistributionMixin<E>::getNormalisationConstant(const EnvironmentType &e,
                                                                                 const StateType &s) const {
  return 1.0F;
}

template <environment::EnvironmentType E>
typename GDT::ActionSpace GreedyDistributionMixin<E>::sampleAction(const EnvironmentType &e, const StateType &s) const {
  return this->getArgmaxAction(e, s);
}

template <objectives::isValueFunctionKeymaker KEYMAPPER_T,
          objectives::isValue VALUE_T,
          auto INITIAL_VALUE = 0.0F,
          auto DISCOUNT_RATE = 0.0F>
struct GreedyPolicy : virtual Policy<typename KEYMAPPER_T::EnvironmentType>,
                      virtual GreedyDistributionMixin<typename KEYMAPPER_T::EnvironmentType>,
                      virtual PolicyValueFunctionMixin<KEYMAPPER_T, VALUE_T, INITIAL_VALUE, DISCOUNT_RATE> {

  SETUP_TYPES_FROM_NESTED_ENVIRON(SINGLE_ARG(KEYMAPPER_T::EnvironmentType));

  ActionSpace operator()(const EnvironmentType &e, const StateType &s) const override;
  //  ActionSpace getArgmaxAction(const EnvironmentType &e, const StateType &s) const = 0;
};

template <objectives::isValueFunctionKeymaker KEYMAPPER_T,
          objectives::isValue VALUE_T,
          auto INITIAL_VALUE,
          auto DISCOUNT_RATE>
typename GPT::ActionSpace GPT::operator()(const EnvironmentType &e, const StateType &s) const {
  return this->getArgmaxAction(e, s);
}

} // namespace policy

#undef GPT