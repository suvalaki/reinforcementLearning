#include "catch.hpp"
#include <iostream>

#include "xtensor/xio.hpp"
#include <xtensor/xrandom.hpp>
// XTensor View:
#include "xtensor/xview.hpp"

#include "markov_decision_process/finite_transition_model.hpp"
#include "policy/random_policy.hpp"
#include "policy/state_action_keymaker.hpp"

TEST_CASE("Finite MDP creating a type works") {

  // simple 2 state model - coin toss
  using CoinSpecComponent = spec::BoundedAarraySpec<float, 0.0F, 1.0F, 1>;
  using CoinSpec = spec::CompositeArraySpec<CoinSpecComponent>;

  using CoinState = state::State<float, CoinSpec>;
  using CoinAction = action::Action<CoinState, CoinSpec>;
  using CoinStep = step::Step<CoinAction>;

  struct CoinReward : reward::Reward<CoinAction> {
    static PrecisionType reward(const TransitionType &t) {
      // return t.nextState.value == 1 ? 1.0F : 0.0F;
      return 1.0F;
    }
  };

  using CoinReturn = returns::Return<CoinReward>;

  using BaseEnviron =
      environment::Environment<CoinStep, CoinReward, CoinReturn>;

  using T = typename BaseEnviron::TransitionType;
  using P = typename BaseEnviron::PrecisionType;

  struct CoinEnviron
      : environment::MarkovDecisionEnvironment<CoinStep, CoinReward,
                                               CoinReturn> {
    using environment::MarkovDecisionEnvironment<
        CoinStep, CoinReward, CoinReturn>::MarkovDecisionEnvironment;
    void reset() override { this->state = CoinState{0.0F, {}}; }
  };

  using CoinTransitionModel = typename CoinEnviron::TransitionModel;

  // Fill out the entire matrix of transition probs
  auto s0 = CoinState{0.0F, {}};
  auto s1 = CoinState{1.0F, {}};
  auto a0 = CoinAction{0};
  auto a1 = CoinAction{1};
  auto transitionModel = CoinTransitionModel{                       //
                                             {T{s0, a0, s0}, 0.8F}, //
                                             {T{s0, a0, s1}, 0.2F}, //
                                             {T{s0, a1, s0}, 0.3F}, //
                                             {T{s0, a1, s1}, 0.7F}, //
                                             {T{s1, a0, s0}, 0.1F}, //
                                             {T{s1, a0, s1}, 0.9F}, //
                                             {T{s1, a1, s0}, 0.5F}, //
                                             {T{s1, a1, s1}, 0.5F}};

  auto environ = CoinEnviron{transitionModel, s0};
}
