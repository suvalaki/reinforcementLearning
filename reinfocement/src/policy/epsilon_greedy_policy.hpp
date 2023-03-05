#pragma once
#include <random>
#include <xtensor/xfixed.hpp>
#include <xtensor/xrandom.hpp>

#include "environment.hpp"
#include "policy/policy.hpp"

#define EGP EpsilonSoftPolicy<EXPLORE_POLICY, EXPLOIT_POLICY, E>

namespace policy {

// The epsilon soft policy is a special type of distribution policy that is a mixture of two other policies. The
// explore policy is used with probability epsilon and the exploit policy is used with probability 1 - epsilon.
// In this way if the exploit policy is deterministic (eg a Greedy policy) then (so long as the explore policy is
// stochastic) the epsilon soft policy is able to continue to explore the environment.

// We say that the charicteristics of the distribution for the epsilon soft policy is the joint distribution
// implicit in the epsilon selection criteria.

template <implementsPolicy EXPLORE_POLICY, implementsPolicy EXPLOIT_POLICY, class E = xt::random::default_engine_type>
requires(std::is_same_v<typename EXPLORE_POLICY::EnvironmentType,
                        typename EXPLOIT_POLICY::EnvironmentType>) struct EpsilonSoftPolicy
    : EXPLOIT_POLICY,
      virtual PolicyDistributionMixin<typename EXPLORE_POLICY::EnvironmentType> {

  SETUP_TYPES_FROM_NESTED_ENVIRON(SINGLE_ARG(EXPLORE_POLICY::EnvironmentType));
  using ExploreType = EXPLORE_POLICY;
  using ExploitType = EXPLOIT_POLICY;
  using EngineType = E;

  ExploreType explorePolicy;
  PrecisionType epsilon = 0.1;
  EngineType &engine;

  EpsilonSoftPolicy(const ExploreType &explorePolicy,
                    const ExploitType &exploitPolicy,
                    PrecisionType epsilon = 0.1,
                    E &engine = xt::random::get_default_random_engine())
      : EXPLOIT_POLICY(exploitPolicy), explorePolicy(explorePolicy), epsilon(epsilon), engine(engine) {}

  ActionSpace explore(const EnvironmentType &e, const StateType &s) const;
  ActionSpace exploit(const EnvironmentType &e, const StateType &s) const;
  ActionSpace operator()(const EnvironmentType &e, const StateType &s) const override;

  PrecisionType getProbability(const EnvironmentType &e, const StateType &s, const ActionSpace &action) const override;
  PrecisionType
  getLogProbability(const EnvironmentType &e, const StateType &s, const ActionSpace &action) const override;
  PrecisionType getKernel(const EnvironmentType &e, const StateType &s, const ActionSpace &action) const override;
  PrecisionType getNormalisationConstant(const EnvironmentType &e, const StateType &s) const override;

  ActionSpace sampleAction(const EnvironmentType &e, const StateType &s) const override;
  ActionSpace getArgmaxAction(const EnvironmentType &e, const StateType &s) const override;
};

template <implementsPolicy EXPLORE_POLICY, implementsPolicy EXPLOIT_POLICY, class E>
typename EGP::ActionSpace EGP::explore(const EnvironmentType &e, const StateType &s) const {
  return this->explorePolicy(e, s);
}

template <implementsPolicy EXPLORE_POLICY, implementsPolicy EXPLOIT_POLICY, class E>
typename EGP::ActionSpace EGP::exploit(const EnvironmentType &e, const StateType &s) const {
  return ExploitType::operator()(e, s);
}

template <implementsPolicy EXPLORE_POLICY, implementsPolicy EXPLOIT_POLICY, class E>
typename EGP::ActionSpace EGP::operator()(const EnvironmentType &e, const StateType &s) const {
  return this->sampleAction(e, s);
}

template <implementsPolicy EXPLORE_POLICY, implementsPolicy EXPLOIT_POLICY, class E>
typename EGP::PrecisionType
EGP::getProbability(const EnvironmentType &e, const StateType &s, const ActionSpace &action) const {

  // The distribution under this policy is epsilon / |A(s)| , where A(s) are actions available at s
  //
  // With probability 1-e we exploit - then the probability given we are exploiting for the action is getProb from
  // exploit With probability e we explore - then the probability given we are exploring for the action is getProb
  // from explore
  //
  // Hence probability of action is:
  // P = (1-e) * P(action | exploit) + e * P(action | explore = (1-e) * getProb(exploit) + e * getProb(explore

  PrecisionType probActionGivenExploit = 1.0F; // When a policy has no distribibution it is deterministic
  PrecisionType probActionGivenExplore = 1.0F;

  if constexpr (implementsPolicyDistributionMixin<ExploitType>) {
    probActionGivenExploit = ExploitType::getProbability(e, s, action);
  }

  if constexpr (implementsPolicyDistributionMixin<ExploreType>) {
    probActionGivenExplore = explorePolicy.getProbability(e, s, action);
  }

  return (1.0F - epsilon) * probActionGivenExploit + epsilon * probActionGivenExplore;
}

template <implementsPolicy EXPLORE_POLICY, implementsPolicy EXPLOIT_POLICY, class E>
typename EGP::PrecisionType
EGP::getLogProbability(const EnvironmentType &e, const StateType &s, const ActionSpace &action) const {
  return std::log(getProbability(e, s, action));
}

template <implementsPolicy EXPLORE_POLICY, implementsPolicy EXPLOIT_POLICY, class E>
typename EGP::PrecisionType
EGP::getKernel(const EnvironmentType &e, const StateType &s, const ActionSpace &action) const {
  throw std::runtime_error("EpsilonSoftPolicy does not have a kernel. You will need to cast this to the appropriate "
                           "ExploreType or Exploit type to index their kernels.");
}

template <implementsPolicy EXPLORE_POLICY, implementsPolicy EXPLOIT_POLICY, class E>
typename EGP::PrecisionType EGP::getNormalisationConstant(const EnvironmentType &e, const StateType &s) const {
  throw std::runtime_error(
      "EpsilonSoftPolicy does not have a normalisation constant. You will need to cast this to the appropriate "
      "ExploreType or Exploit type to index their normalisation constants.");
}

template <implementsPolicy EXPLORE_POLICY, implementsPolicy EXPLOIT_POLICY, class E>
typename EGP::ActionSpace EGP::sampleAction(const EnvironmentType &e, const typename EGP::StateType &s) const {
  if (xt::random::rand<double>(xt::xshape<1>{}, 0, 1, engine)[0] < epsilon) {
    return this->explore(e, s);
  }
  return this->exploit(e, s);
}

template <implementsPolicy EXPLORE_POLICY, implementsPolicy EXPLOIT_POLICY, class E>
typename EGP::ActionSpace EGP::getArgmaxAction(const EnvironmentType &e, const StateType &s) const {

  // TODO?: Make this robust to any epsilon and pick the actual action with biggest probability between the two policies

  if constexpr (implementsPolicyDistributionMixin<typename EGP::ExploitType>) {
    return ExploitType::getArgmaxAction(e, s);
  }
  return this->exploit(e, s);
}

} // namespace policy

#undef EGP