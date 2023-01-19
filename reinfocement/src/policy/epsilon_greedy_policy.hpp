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
  using baseType = EXPLOIT_POLICY;
  using EnvironmentType = typename baseType::EnvironmentType;
  using StateType = typename baseType::StateType;
  using ActionSpace = typename baseType::ActionSpace;
  using TransitionType = typename baseType::TransitionType;
  using RewardType = typename EnvironmentType::RewardType;
  using PrecisionType = typename RewardType::PrecisionType;

  RandomPolicy<ENVIRON_T> randomPolicy;

  PrecisionType epsilon = 0.1;
  E &engine;

  EpsilonGreedyPolicy(PrecisionType epsilon = 0.1,
                      E &engine = xt::random::get_default_random_engine())
      : epsilon(epsilon), engine(engine) {}

  ActionSpace explore(const StateType &s) { return randomPolicy(s); }

  ActionSpace exploit(const StateType &s) {
    return static_cast<EXPLOIT_POLICY &>(*this)(s);
  }

  ActionSpace operator()(const StateType &s) override {
    if (xt::random::rand<double>(xt::xshape<1>{}, 0, 1, engine)[0] < epsilon) {
      return explore(s);
    } else {
      return exploit(s);
    }
  }
};

} // namespace policy