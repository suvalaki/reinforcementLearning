#include "catch.hpp"
#include <cmath>
#include <iostream>

#include "environment_fixtures.hpp"
#include "markov_decision_process/coin_mdp.hpp"
#include "monte_carlo/value.hpp"
#include "monte_carlo/value_update/ordinary_importance_sampling.hpp"

using namespace monte_carlo;

struct MockPolicyWithFixedRatio : CoinDistributionPolicy {
  using Base = CoinDistributionPolicy;
  using Base::Base;
  PrecisionType importanceSamplingRatio(const EnvironmentType &environment,
                                        const StateType &state,
                                        const ActionSpace &action,
                                        const Base &other) const {
    return 2;
  }
};

TEST_CASE("monte_carlo::OrdinaryImportanceSamplingUpdate") {
  auto data = CoinModelDataFixture{};
  auto &[s0, s1, a0, a1, transitionModel, environ, policy, policyState, policyAction, _v0, valueFunction, _v2] = data;

  auto mockPolicy = MockPolicyWithFixedRatio();

  auto updater = monte_carlo::OrdinaryImportanceSamplingUpdate<std::decay_t<decltype(valueFunction)>>();
  // Average returns should still work when empty
  updater.updateReturns(valueFunction, mockPolicy, mockPolicy, environ, s0, a0, 1);
  auto key = decltype(updater)::KeyMaker::make(environ, s0, a0);
  auto averageReturn0 = updater.getWeightedReturn(key);
  REQUIRE(averageReturn0 == 2); // the return (1) * the importance sampling ratio (2)
  updater.updateValue(valueFunction, mockPolicy, mockPolicy, environ, s0, a0);
  REQUIRE(valueFunction[key].value == 2);

  updater.updateReturns(valueFunction, mockPolicy, mockPolicy, environ, s0, a0, 2);
  auto averageReturn1 = updater.getWeightedReturn(key);
  // (2 + (2 * 2 * 2)) / 2    (old + (new * importance sampling ratio)) / size
  REQUIRE(averageReturn1 == 5);
  updater.updateValue(valueFunction, mockPolicy, mockPolicy, environ, s0, a0);
  REQUIRE(valueFunction[key].value == 5);
}

TEST_CASE("monte_carlo::OrdinaryImportanceSamplingIncrementalUpdate") {
  auto data = CoinModelDataFixture{};
  auto &[s0, s1, a0, a1, transitionModel, environ, policy, policyState, policyAction, _v0, valueFunction, _v2] = data;

  auto mockPolicy = MockPolicyWithFixedRatio();

  auto updater = monte_carlo::OrdinaryImportanceSamplingIncrementalUpdate<std::decay_t<decltype(valueFunction)>>();
  // Average returns should still work when empty
  updater.updateReturns(valueFunction, mockPolicy, mockPolicy, environ, s0, a0, 1);
  auto key = decltype(updater)::KeyMaker::make(environ, s0, a0);
  REQUIRE(updater.returns[key].averageWeightedReturn == 2);
  updater.updateValue(valueFunction, mockPolicy, mockPolicy, environ, s0, a0);
  REQUIRE(valueFunction[key].value == 2);

  updater.updateReturns(valueFunction, mockPolicy, mockPolicy, environ, s0, a0, 2);
  REQUIRE(updater.returns[key].averageWeightedReturn == 5);
  updater.updateValue(valueFunction, mockPolicy, mockPolicy, environ, s0, a0);
  REQUIRE(valueFunction[key].value == 5);
}
