#include "catch.hpp"
#include <cmath>
#include <iostream>

#include "environment_fixtures.hpp"
#include "markov_decision_process/coin_mdp.hpp"
#include "monte_carlo/value.hpp"
#include "monte_carlo/value_update/weighted_importance_sampling.hpp"

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

TEST_CASE("monte_carlo::WeightedImportanceSamplingUpdate") {
  auto data = CoinModelDataFixture{};
  auto &[s0, s1, a0, a1, transitionModel, environ, policy, policyState, policyAction, _v0, valueFunction, _v2] = data;

  auto mockPolicy = MockPolicyWithFixedRatio();

  auto updater = monte_carlo::WeightedImportanceSamplingUpdate<std::decay_t<decltype(valueFunction)>>();
  // Average returns should still work when empty
  updater.updateReturns(valueFunction, mockPolicy, mockPolicy, environ, s0, a0, 1);
  auto key = decltype(updater)::KeyMaker::make(environ, s0, a0);
  auto averageReturn0 = updater.getWeightedReturn(key);
  REQUIRE(averageReturn0 == 1); // the return (1) * the importance sampling ratio (2) / importance sampling ratio (2)
  updater.updateValue(valueFunction, mockPolicy, mockPolicy, environ, s0, a0);
  REQUIRE(valueFunction[key].value == 1);

  updater.updateReturns(valueFunction, mockPolicy, mockPolicy, environ, s0, a0, 2);
  auto averageReturn1 = updater.getWeightedReturn(key);
  //  (r0 * i0 + r1 * i0 * i1) / (i0 + i1) = (2 * 2 + 2 * (1 * 2)) / (2 + 2 * 2)
  REQUIRE(averageReturn1 == Approx((1 * 2 + (2 * 2 * 2)) / 6.0f));
  updater.updateValue(valueFunction, mockPolicy, mockPolicy, environ, s0, a0);
  REQUIRE(valueFunction[key].value == Approx((1 * 2 + (2 * 2 * 2)) / 6.0f));
}

TEST_CASE("monte_carlo::WeightedImportanceSamplingIncrementalUpdate") {
  auto data = CoinModelDataFixture{};
  auto &[s0, s1, a0, a1, transitionModel, environ, policy, policyState, policyAction, _v0, valueFunction, _v2] = data;

  auto mockPolicy = MockPolicyWithFixedRatio();

  auto updater = monte_carlo::WeightedImportanceSamplingIncrementalUpdate<std::decay_t<decltype(valueFunction)>>();
  // Average returns should still work when empty
  updater.updateReturns(valueFunction, mockPolicy, mockPolicy, environ, s0, a0, 1);
  auto key = decltype(updater)::KeyMaker::make(environ, s0, a0);
  REQUIRE(updater.returns[key].averageWeightedReturn == 1);
  updater.updateValue(valueFunction, mockPolicy, mockPolicy, environ, s0, a0);
  REQUIRE(valueFunction[key].value == 1);

  updater.updateReturns(valueFunction, mockPolicy, mockPolicy, environ, s0, a0, 2);
  REQUIRE(updater.returns[key].averageWeightedReturn == Approx((1 * 2 + (2 * 2 * 2)) / 6.0f));
  updater.updateValue(valueFunction, mockPolicy, mockPolicy, environ, s0, a0);
  REQUIRE(valueFunction[key].value == Approx((1 * 2 + (2 * 2 * 2)) / 6.0f));
}
