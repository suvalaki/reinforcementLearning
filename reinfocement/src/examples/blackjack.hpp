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
  ACE = 1,
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

int cardToVal(const Cards &c) {
  switch (c) {
  case Cards::ACE:
    return 1;
  case Cards::TWO:
    return 2;
  case Cards::THREE:
    return 3;
  case Cards::FOUR:
    return 4;
  case Cards::FIVE:
    return 5;
  case Cards::SIX:
    return 6;
  case Cards::SEVEN:
    return 7;
  case Cards::EIGHT:
    return 8;
  case Cards::NINE:
    return 9;
  case Cards::TEN:
    return 10;
  case Cards::JACK:
    return 10;
  case Cards::QUEEN:
    return 10;
  case Cards::KING:
    return 10;
  }
  return 0;
}

// Each player is limited to drawing 10 cards. - Let us also assume a single player
// looking to get as close to 21 as possible
constexpr std::size_t max_cards = 11;
constexpr std::size_t players = 1;
constexpr std::size_t n_cards = 14; // need one extra

// We need to mark the visible cards available to each player.
// In actual blackjack the dealer has one face up card and one face down card. This is a simplification with 2 faceup
// cards
using BlackjackPlayerStateSpec = spec::BoundedAarraySpec<int, 2, 22, 1>;
using BlackjackHasAceSpec = spec::BoundedAarraySpec<int, 0, 2, 1>;
using BlackjackDealerStateSpec = spec::BoundedAarraySpec<int, 2, 22, 1>;
using BlackjackStateSpec = spec::
    CompositeArraySpec<BlackjackPlayerStateSpec, BlackjackHasAceSpec, BlackjackDealerStateSpec, BlackjackHasAceSpec>;
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

  static int bestPossibleScore(
      const typename BlackjackPlayerStateSpec::DataType &handState,
      const typename BlackjackHasAceSpec::DataType &hasAce) {
    auto handVal = handState.at(0);
    if (hasAce.at(0) == 1) {
      if (handVal + 10 <= 21) {
        return handVal + 10;
      }
    }
    return handVal;
  }

  static PrecisionType reward(const TransitionType &t) {

    const auto playerScore =
        bestPossibleScore(std::get<0>(t.nextState.observable), std::get<1>(t.nextState.observable));
    const auto dealerScore =
        bestPossibleScore(std::get<2>(t.nextState.observable), std::get<3>(t.nextState.observable));

    if (dealerScore > 21)
      return 0.5;

    if (playerScore > 21 && dealerScore <= 21)
      return 0.0F;

    if (playerScore > dealerScore && playerScore <= 21 && dealerScore <= 21)
      return 1.0F;

    return 0.0F;
  }
};

using BlackjackReturn = returns::Return<BlackjackReward>;

constexpr auto maxVal = 21; // cant go above this value
constexpr auto nHasAce = 2;
constexpr auto nStates = maxVal * nHasAce * maxVal * nHasAce;
constexpr auto nActions = 2;

template <environment::RewardType REWARD_T, environment::ReturnType RETURN_T>
struct BlackjackEnvironment : environment::FiniteEnvironment<BlackjackStep, REWARD_T, RETURN_T> {

  SETUP_TYPES_W_ENVIRON(
      SINGLE_ARG(environment::Environment<BlackjackStep, REWARD_T, RETURN_T>), SINGLE_ARG(BlackjackEnvironment));

  // init random device
  std::random_device rd;
  std::mt19937 gen{rd()};

  Cards dealerDraw() {
    std::uniform_int_distribution<> dis(1, 13);
    return Cards{dis(gen)};
  }

  StateType stateFromIndex(std::size_t index) const override {
    auto i = index / (nHasAce * maxVal * nHasAce);
    auto j = (index - i * nHasAce * maxVal * nHasAce) / (maxVal * nHasAce);
    auto k = (index - i * nHasAce * maxVal * nHasAce - j * maxVal * nHasAce) / nHasAce;
    auto l = index - i * nHasAce * maxVal * nHasAce - j * maxVal * nHasAce - k * nHasAce;
    return StateType({{static_cast<int>(i)}, {static_cast<int>(j)}, {static_cast<int>(k)}, {static_cast<int>(l)}}, {});
  };

  ActionSpace actionFromIndex(std::size_t index) const override { return ActionSpace(static_cast<int>(index)); };

  std::unordered_set<StateType, typename StateType::Hash> getAllPossibleStates() const override {
    auto states = std::unordered_set<StateType, typename StateType::Hash>{};
    for (auto i = 0; i < maxVal; ++i)
      for (auto j = 0; j < nHasAce; ++j)
        for (auto k = 0; k < maxVal; ++k)
          for (auto l = 0; l < nHasAce; ++l)
            states.emplace(StateType(
                {{static_cast<int>(i)}, {static_cast<int>(j)}, {static_cast<int>(k)}, {static_cast<int>(l)}}, {}));
    return states;
  };

  std::unordered_set<ActionSpace, typename ActionSpace::Hash> getAllPossibleActions() const {
    auto actions = std::unordered_set<ActionSpace, typename ActionSpace::Hash>{};
    for (auto i = 0; i < nActions; ++i)
      actions.emplace(ActionSpace(i));
    return actions;
  };

  StateType reset() override {

    auto observable = ::policy::random_spec_gen<BlackjackStateSpec>(rd);

    // Enforce the dealer strategy of Required to hit on 16 or less

    this->state = StateType(observable, {});
    return this->state;
  };

  bool gameContinues(const TransitionType &t) {
    return (std::get<0>(t.nextState.observable).at(0) < 21) and (t.action.playerAction() == BlackjackChoices::HIT);
  }

  void applyDealerStrategy(TransitionType &t) {
    while (true) {
      auto &dealerVal = std::get<2>(t.nextState.observable).at(0);
      auto &dealerAce = std::get<3>(t.nextState.observable).at(0);
      auto dealerScore = RewardType::bestPossibleScore(dealerVal, dealerAce);

      // Have the dealer hit until he has 17 or more
      if (dealerScore >= 17)
        break;

      auto dealerCard = dealerDraw();

      if (dealerCard == Cards::ACE) {
        std::get<3>(t.nextState.observable).at(0) = 1;
      }
      std::get<2>(t.nextState.observable).at(0) += cardToVal(dealerCard);
    }
  }

  TransitionType step(const ActionSpace &action) override {

    auto nextState = this->state;
    if (action.playerAction() == BlackjackChoices::HIT) {
      // Draw a card from the deck and add it to the players hand
      auto card = dealerDraw();
      if (card == Cards::ACE) {
        std::get<1>(nextState.observable) = 1;
      }
      std::get<0>(nextState.observable).at(0) += cardToVal(card);
    }

    auto t = TransitionType{this->state, action, nextState};
    if (not gameContinues(t)) {
      t.kind = environment::TransitionKind::TERMINAL;
      applyDealerStrategy(t);
    }
    return t;
  }

  StateType getNullState() const override { return StateType(); };
};

} // namespace examples::blackjack
