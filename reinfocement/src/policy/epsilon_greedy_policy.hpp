#pragma once
#include <random>
#include <xtensor/xfixed.hpp>
#include <xtensor/xrandom.hpp>

#include "action.hpp"
#include "environment.hpp"
#include "spec.hpp"

#include "greedy_policy.hpp"
#include "policy.hpp"
#include "random_policy.hpp"

namespace policy {

using spec::CompositeArraySpecType;
using spec::isBoundedArraySpec;
using spec::isCategoricalArraySpec;

template <environment::EnvironmentType ENVIRON_T,
          PolicyType EXPLOIT_POLICY = GreedyPolicy<ENVIRON_T>,
          class E = xt::random::default_engine_type>
struct EpsilonGreedyPolicy : EXPLOIT_POLICY {

  SETUP_TYPES(SINGLE_ARG(EXPLOIT_POLICY));
  using EnvironmentType = typename BaseType::EnvironmentType;
  using KeyMaker = typename EXPLOIT_POLICY::KeyMaker;
  using KeyType = typename EXPLOIT_POLICY::KeyType;

  RandomPolicy<ENVIRON_T> randomPolicy;
  PrecisionType epsilon = 0.1;
  E &engine;

  EpsilonGreedyPolicy(PrecisionType epsilon = 0.1, E &engine = xt::random::get_default_random_engine())
      : epsilon(epsilon), engine(engine) {}

  ActionSpace explore(const StateType &s) { return randomPolicy(s); }

  ActionSpace exploit(const StateType &s) const {
    return const_cast<EXPLOIT_POLICY &>(static_cast<const EXPLOIT_POLICY &>(*this))(s);
  }

  ActionSpace operator()(const StateType &s) override {
    if (xt::random::rand<double>(xt::xshape<1>{}, 0, 1, engine)[0] < epsilon) {
      return explore(s);
    } else {
      return exploit(s);
    }
  }
  ActionSpace operator()(const EnvironmentType &e, const StateType &s) { return (*this)(s); }

  PrecisionType getProbability(const EnvironmentType &e, const StateType &s, const KeyType &key) const {
    // The distribution under this policy is epsilon / |A(s)| , where A(s) are actions available at s
    auto exploitAction = exploit(s);
    auto action = KeyMaker::get_action_from_key(key);
    auto nActions = e.getReachableActions(s).size();
    if (exploitAction == action)
      return 1.0F - epsilon + epsilon / nActions;
    return epsilon / nActions;
  }
};

} // namespace policy