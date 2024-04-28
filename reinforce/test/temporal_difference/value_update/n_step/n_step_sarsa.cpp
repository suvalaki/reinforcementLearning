#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <iostream>

#include <reinforce/temporal_difference/value_update/n_step/n_step_sarsa.hpp>

#include "markov_decision_process/coin_mdp.hpp"

using namespace temporal_difference;

TEST_CASE("temporal_difference::NStepSarsa") {

  const std::size_t n = 10;
  const std::size_t t = 5;
  const std::size_t max_steps = 25;

  auto data = CoinModelDataFixture{};
  auto &[s0, s1, a0, a1, transitionModel, environ, policySA, policyState, policyAction, valueFunction, _v1, _v2] = data;

  auto updater = NStepSarsaUpdater<std::decay_t<decltype(valueFunction)>>(10);

  const auto s = updater.step(valueFunction, policySA, policySA, environ, a0, 1.0F);
  const auto a = updater.store(valueFunction, policySA, policySA, environ, 1.0F, n, 1, s);

  using BufferType = typename decltype(updater)::BufferType;
  using BufferAtom = typename decltype(updater)::ExpandedStatefulUpdateResult;
  auto buffer = BufferType{10};
  auto atom = BufferAtom{s, a};
  buffer.push_back(atom);
  buffer.push_back(atom);
  buffer.push_back(atom);

  std::size_t T = 10;
  updater.observe(valueFunction, policySA, policySA, environ, a0, 1.0F, n, t, buffer);

  auto b = buffer.begin() + 1;

  auto g = updater.calculateReturn(valueFunction, policySA, policySA, environ, 1.0F, b, b);

  // singular update
  updater.update(valueFunction, policySA, policySA, environ, a0, 1.0F, n, t, buffer);

  // batch update
  updater.update(valueFunction, policySA, policySA, environ, 1.0F, max_steps);
}

TEST_CASE("temporal_difference::NStepSarsaOffPolicy") {

  const std::size_t n = 10;
  const std::size_t t = 5;
  const std::size_t max_steps = 25;

  auto data = CoinModelDataFixture{};
  auto &[s0, s1, a0, a1, transitionModel, environ, policySA, policyState, policyAction, valueFunction, _v1, _v2] = data;

  auto updater = NStepSarsaOffPolicyUpdater<std::decay_t<decltype(valueFunction)>>(10);

  const auto s = updater.step(valueFunction, policySA, policySA, environ, a0, 1.0F);
  const auto a = updater.store(valueFunction, policySA, policySA, environ, 1.0F, n, 1, s);

  using BufferType = typename decltype(updater)::BufferType;
  using BufferAtom = typename decltype(updater)::ExpandedStatefulUpdateResult;
  auto buffer = BufferType{10};
  auto atom = BufferAtom{s, a};
  buffer.push_back(atom);
  buffer.push_back(atom);
  buffer.push_back(atom);

  // std::cout << "here: " << buffer.size() << std::endl;

  std::size_t T = 10;
  updater.observe(valueFunction, policySA, policySA, environ, a0, 1.0F, n, t, buffer);

  auto b = buffer.begin() + 1;

  auto g = updater.calculateReturn(valueFunction, policySA, policySA, environ, 1.0F, b, b);

  // singular update
  updater.update(valueFunction, policySA, policySA, environ, a0, 1.0F, n, t, buffer);

  // batch update
  updater.update(valueFunction, policySA, policySA, environ, 1.0F, max_steps);
}