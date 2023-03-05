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
#define GPT GreedyPolicy<VALUE_FUNCTION_T>

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

template <objectives::isValueFunction VALUE_FUNCTION_T>
struct GreedyPolicy : virtual Policy<typename VALUE_FUNCTION_T::EnvironmentType>,
                      virtual GreedyDistributionMixin<typename VALUE_FUNCTION_T::EnvironmentType> {

  SETUP_TYPES_FROM_NESTED_ENVIRON(SINGLE_ARG(VALUE_FUNCTION_T::EnvironmentType));
  // using ValueFunctionBaseType = VALUE_FUNCTION_T;

  ActionSpace operator()(const EnvironmentType &e, const StateType &s) const override;
  //  ActionSpace getArgmaxAction(const EnvironmentType &e, const StateType &s) const = 0;

  GreedyPolicy(auto &&...args) {}
  GreedyPolicy(const GreedyPolicy &p) {}
  GreedyPolicy &operator=(GreedyPolicy &&g) {
    VALUE_FUNCTION_T(std::move(g));
    return *this;
  }
};

template <objectives::isValueFunction VALUE_FUNCTION_T>
typename GPT::ActionSpace GPT::operator()(const EnvironmentType &e, const StateType &s) const {
  return this->getArgmaxAction(e, s);
}

} // namespace policy

#undef GPT