#pragma once

#include <iostream>
#include <tuple>
#include <type_traits>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>

#include <reinforce/markov_decision_process/finite_transition_model.hpp>
#include <reinforce/policy/distribution_policy.hpp>
#include <reinforce/policy/finite/distribution_policy.hpp>
#include <reinforce/policy/random_policy.hpp>
#include <reinforce/policy/state_action_value.hpp>
#include <reinforce/policy/value.hpp>

// simple 2 state model - coin toss
enum e0 { HEADS, TAILS };
using CoinSpecComponent = spec::CategoricalArraySpec<e0, 2, 1>;

static_assert(spec::isCategoricalArraySpec<CoinSpecComponent>, "SPEC FAILED");

using CoinSpec = spec::CompositeArraySpec<CoinSpecComponent>;

using CoinState = state::State<float, CoinSpec>;
using CoinAction = action::Action<CoinState, CoinSpec>;
using CoinStep = step::Step<CoinAction>;

struct CoinReward : reward::Reward<CoinAction> {
  static PrecisionType reward(const TransitionType &t) {
    // return t.nextState.value == 1 ? 1.0F : 0.0F;
    return std::get<0>(t.nextState.observable)[0] == 1 ? 1.0 : -1.0;
  }
};

using CoinReturn = returns::Return<CoinReward>;
using DiscountCoinReturn = returns::DiscountReturn<CoinReward, 0.5F>;

using BaseEnviron = environment::FiniteEnvironment<CoinStep, CoinReward, CoinReturn>;

using T = typename BaseEnviron::TransitionType;
using P = typename BaseEnviron::PrecisionType;

static CoinState s0 = CoinState{0.0F, {}};
static CoinState s1 = CoinState{1.0F, {}};

struct CoinEnviron : environment::MarkovDecisionEnvironment<CoinStep, CoinReward, DiscountCoinReturn> {

  SETUP_TYPES(SINGLE_ARG(environment::MarkovDecisionEnvironment<CoinStep, CoinReward, DiscountCoinReturn>));
  // using EnvironmentType = typename EnvironmentType::EnvironmentType;
  using BaseType::BaseType;

  StateType reset() override {
    this->state = CoinState{0.0F, {}};
    return this->state;
  }
  StateType getNullState() const override { return CoinState{0.0F, {}}; }
};

static_assert(environment::FiniteEnvironmentType<CoinEnviron>, "ENVIRON FAILED");

using CoinTransitionModel = typename CoinEnviron::TransitionModel;

using CoinFiniteStateActionValueFunction = policy::objectives::FiniteStateActionValueFunction<CoinEnviron>;
using CoinFiniteStateValueFunction = policy::objectives::FiniteStateValueFunction<CoinEnviron>;
using CoinFiniteActionValueFunction = policy::objectives::FiniteActionValueFunction<CoinEnviron>;

using CoinDistributionPolicy = policy::FiniteDistributionPolicy<CoinFiniteStateActionValueFunction>;
using CoinDistributionPolicyState = policy::FiniteDistributionPolicy<CoinFiniteStateValueFunction>;
using CoinDistributionPolicyAction = policy::FiniteDistributionPolicy<CoinFiniteActionValueFunction>;

struct CoinModelDataFixture {

  CoinState s0 = CoinState{0.0F, {}};
  CoinState s1 = CoinState{1.0F, {}};
  CoinAction a0 = CoinAction{0};
  CoinAction a1 = CoinAction{1};
  CoinTransitionModel transitionModel = CoinTransitionModel{
      .transitions =
          typename CoinTransitionModel::TransitionModelMap{
              {T{s0, a0, s0}, 0.8F}, //
              {T{s0, a0, s1}, 0.2F}, //
              {T{s0, a1, s0}, 0.3F}, //
              {T{s0, a1, s1}, 0.7F}, //
              {T{s1, a0, s0}, 0.1F}, //
              {T{s1, a0, s1}, 0.9F}, //
              {T{s1, a1, s0}, 0.5F}, //
              {T{s1, a1, s1}, 0.5F}  //
          },
      .states = {s0, s1}, //
      .actions = {a0, a1} //
  };
  CoinEnviron environ = CoinEnviron{transitionModel, s0};
  CoinDistributionPolicy policy = CoinDistributionPolicy{};
  CoinDistributionPolicyState policyState;
  CoinDistributionPolicyAction policyAction;
  CoinFiniteStateActionValueFunction valueFunctionStateAction = CoinFiniteStateActionValueFunction{};
  CoinFiniteStateValueFunction valueFunctionState = CoinFiniteStateValueFunction{};
  CoinFiniteActionValueFunction valueFunctionAction = CoinFiniteActionValueFunction{};

  CoinModelDataFixture() { policy.initialize(environ); }

  using TypeList = std::tuple<
      CoinState,
      CoinState,
      CoinAction,
      CoinAction,
      CoinTransitionModel,
      CoinEnviron,
      CoinDistributionPolicy,
      CoinDistributionPolicyState,
      CoinDistributionPolicyAction,
      CoinFiniteStateActionValueFunction,
      CoinFiniteStateValueFunction,
      CoinFiniteActionValueFunction>;

  constexpr static std::size_t typeListSize = std::tuple_size_v<typename CoinModelDataFixture::TypeList>;

  template <std::size_t Index>
  auto &&get() & {
    return get_helper<Index>(*this);
  }

  template <std::size_t Index>
  auto &&get() && {
    return get_helper<Index>(*this);
  }

  template <std::size_t Index>
  auto &&get() const & {
    return get_helper<Index>(*this);
  }

  template <std::size_t Index>
  auto &&get() const && {
    return get_helper<Index>(*this);
  }

private:
  template <std::size_t Index, typename T>
  auto &&get_helper(T &&t) {
    static_assert(Index < typeListSize, "Index out of bounds for CoinModelDataFixture");
    if constexpr (Index == 0)
      return std::forward<T>(t).s0;
    if constexpr (Index == 1)
      return std::forward<T>(t).s1;
    if constexpr (Index == 2)
      return std::forward<T>(t).a0;
    if constexpr (Index == 3)
      return std::forward<T>(t).a1;
    if constexpr (Index == 4)
      return std::forward<T>(t).transitionModel;
    if constexpr (Index == 5)
      return std::forward<T>(t).environ;
    if constexpr (Index == 6)
      return std::forward<T>(t).policy;
    if constexpr (Index == 7)
      return std::forward<T>(t).policyState;
    if constexpr (Index == 8)
      return std::forward<T>(t).policyAction;
    if constexpr (Index == 9)
      return std::forward<T>(t).valueFunctionStateAction;
    if constexpr (Index == 10)
      return std::forward<T>(t).valueFunctionState;
    if constexpr (Index == 11)
      return std::forward<T>(t).valueFunctionAction;
  }
};

static_assert(environment::FullyKnownFiniteStateEnvironment<CoinEnviron>);
static_assert(environment::FullyKnownConditionalStateActionEnvironment<CoinEnviron>);

namespace std {
template <>
struct tuple_size<::CoinModelDataFixture> : integral_constant<size_t, CoinModelDataFixture::typeListSize> {};

template <std::size_t N>
struct tuple_element<N, CoinModelDataFixture> {
  using type = decltype(std::declval<CoinModelDataFixture>().get<N>());
};

} // namespace std