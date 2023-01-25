#pragma once

#include <iostream>

#include "xtensor/xio.hpp"
#include <xtensor/xrandom.hpp>
// XTensor View:
#include "xtensor/xview.hpp"

#include "markov_decision_process/finite_transition_model.hpp"
#include "policy/random_policy.hpp"
#include "policy/state_action_keymaker.hpp"

// simple 2 state model - coin toss
enum e0 { HEADS, TAILS };
using CoinSpecComponent = spec::CategoricalArraySpec<e0, 2, 1>;
using CoinSpec = spec::CompositeArraySpec<CoinSpecComponent>;

using CoinState = state::State<float, CoinSpec>;
using CoinAction = action::Action<CoinState, CoinSpec>;
using CoinStep = step::Step<CoinAction>;

struct CoinReward : reward::Reward<CoinAction> {
  static PrecisionType reward(const TransitionType &t) {
    // return t.nextState.value == 1 ? 1.0F : 0.0F;
    return std::get<0>(t.nextState.observable)[0] == 1 ? 1.0 : 0.0;
  }
};

using CoinReturn = returns::Return<CoinReward>;

using BaseEnviron = environment::Environment<CoinStep, CoinReward, CoinReturn>;

using T = typename BaseEnviron::TransitionType;
using P = typename BaseEnviron::PrecisionType;

struct CoinEnviron
    : environment::MarkovDecisionEnvironment<CoinStep, CoinReward, CoinReturn> {
  using environment::MarkovDecisionEnvironment<
      CoinStep, CoinReward, CoinReturn>::MarkovDecisionEnvironment;
  void reset() override { this->state = CoinState{0.0F, {}}; }
};

using CoinTransitionModel = typename CoinEnviron::TransitionModel;