#pragma once
#include <array>

#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

#include "environment.hpp"
#include "policy/random_policy.hpp"

namespace examples::blackjack {

// In blackjack each player starts with 2 cards and can hit or stand.
// Let us assume this is 2 players. one agent and the dealer.
//
// Let us assume that the cards are ordered 1 to 52 and the value of
// the states are associated with card types.

// In this simplificaction of blackjack the suit doesnt really matter
// we are assuming an infinite deck where the chance of ever card is
// 1/52. The point addition of each card is associated with the number
// of the card in the enum
enum class Cards {
  HIDDEN = 0,
  // HEARTS
  ONE,
  TWO,
  THREE,
  FOUR,
  FIVE,
  SIX,
  SEVEN,
  EIGHT,
  NINE,
  TEN,
  JACK,
  QUEEN,
  KING,
};

// Each player is limited to drawing 10 cards. - Let us also assume a single player
// looking to get as close to 21 as possible
constexpr std::size_t max_cards = 11;
constexpr std::size_t players = 1;
constexpr std::size_t n_cards = 14; // need one extra

// We need to mark the visible cards available to each player.
using BlackjackPlayerStateSpec = spec::BoundedAarraySpec<int, 2, 22, 1>;
using BlackjackHasAceSpec = spec::BoundedAarraySpec<int, 0, 2, 1>;
using BlackjackDealerStateSpec = spec::BoundedAarraySpec<int, 2, 22, 1>;
using BlackjackStateSpec =
    spec::CompositeArraySpec<BlackjackPlayerStateSpec, BlackjackHasAceSpec, BlackjackDealerStateSpec>;
using BlackjackState = environment::State<float, BlackjackStateSpec>;

// using SinglePlayerCardSpec

enum class BlackjackChoices { STAY = 0, HIT };

using BlackjackPlayerActionSpec = spec::CategoricalArraySpec<BlackjackChoices, 2, 1>;
using BlackjackActionSpec = spec::CompositeArraySpec<BlackjackPlayerActionSpec>;

struct BlackjackAction : environment::Action<BlackjackState, BlackjackActionSpec> {
  using environment::Action<BlackjackState, BlackjackActionSpec>::Action;
  BlackjackChoices playerAction() const { return BlackjackChoices{std::get<0>(*this).at(0)}; }
};

using BlackjackStep = environment::Step<BlackjackAction>;

struct BlackjackReward : reward::Reward<BlackjackAction> {

  static PrecisionType reward(const TransitionType &t) {

    // If the player has an ace
    auto handVal = std::get<0>(t.nextState.observable).at(0);
    auto dealerVal = std::get<2>(t.nextState.observable).at(0);
    if (handVal > 21 and dealerVal <= 21) {
      return 0.0F;
    }
    if (std::get<1>(t.nextState.observable).at(0) == 1) {
      if (handVal + 10 <= 21) {
        if (handVal + 10 >= dealerVal)
          return 1.0F;
        return 0.0F;
      }
    }
    if (handVal >= dealerVal)
      return 1.0F;
    return 0.0F;
  }
};

using BlackjackReturn = returns::Return<BlackjackReward>;

template <environment::RewardType REWARD_T, environment::ReturnType RETURN_T>
struct BlackjackEnvironment : environment::Environment<BlackjackStep, REWARD_T, RETURN_T> {

  SETUP_TYPES_W_ENVIRON(SINGLE_ARG(environment::Environment<BlackjackStep, REWARD_T, RETURN_T>),
                        SINGLE_ARG(BlackjackEnvironment));

  // init random device
  std::random_device rd;
  std::mt19937 gen{rd()};

  int dealerDraw() {
    std::uniform_int_distribution<> dis(1, 13);
    return dis(gen);
  }

  StateType reset() override {

    auto observable = ::policy::random_spec_gen<BlackjackStateSpec>();

    // Enforce the dealer strategy of Required to hit on 16 or less

    this->state = StateType(observable, {});
    return this->state;
  };

  bool gameContinues(const TransitionType &t) {
    return (std::get<0>(t.nextState.observable).at(0)) and (t.action.playerAction() == BlackjackChoices::HIT);
  }

  TransitionType step(const ActionSpace &action) override {

    auto nextState = this->state;
    if (action.playerAction() == BlackjackChoices::HIT) {
      // Draw a card from the deck and add it to the players hand
      auto card = dealerDraw();
      if (card == 1) {
        std::get<0>(nextState.observable).at(0) += 1;
        std::get<1>(nextState.observable) = 1;
      }
    }

    auto t = TransitionType{this->state, action, nextState};
    if (not gameContinues(t))
      t.kind = environment::TransitionKind::TERMINAL;
    return t;
  }

  StateType getNullState() const override { return StateType(); };
};

} // namespace examples::blackjack
