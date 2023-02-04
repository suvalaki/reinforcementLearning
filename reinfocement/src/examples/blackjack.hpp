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
  using VisibleCardSpec                 = spec::CategoricalArraySpec<Cards, 10, players>;
  using BlackjackPlayerStateSpec        = spec::CategoricalArraySpec<Cards, n_cards, players, max_cards>;
  using BlackjackStateSpec              = spec::CompositeArraySpec<
                                            BlackjackPlayerStateSpec, VisibleCardSpec 
                                          >;
  using HiddenBlackjackPlayerStateSpec  = spec::CategoricalArraySpec<Cards, n_cards, players, max_cards>;
  using HiddenStateSpec                 = spec::CompositeArraySpec<HiddenBlackjackPlayerStateSpec>;

  struct BlackjackState : environment::State<float, BlackjackStateSpec, HiddenStateSpec> {

    using environment::State<float, BlackjackStateSpec, HiddenStateSpec>::State;

    template <std::size_t player> requires (player < players)
    std::size_t calculateHandValue() const {
      // TODO: Handle ace case
      auto v = xt::view(std::get<0>(observable), player, xt::all());
      auto val = static_cast<std::size_t>(xt::sum(v)[0]);
      auto nAces = static_cast<std::size_t>(xt::sum(xt::where(v<2, v, 0* v ))[0]);

      for (std::size_t i=0; i<nAces ; i++){
        if(val + 10 > 21){
          return val;
        }
        val += 10;
      }
      return val;

    }

    template <std::size_t ...player>
    void calculateHandValues_impl(
        std::array<std::size_t, players>& vals, std::index_sequence<player...>) const {
      ((vals[player] = calculateHandValue<player>()),...);
    }

    std::array<std::size_t, players> calculateHandValues() const  {
      std::array< std::size_t, players> scores;
      calculateHandValues_impl(scores, std::make_index_sequence<players>{});
      return scores;
    }

  };

  //using SinglePlayerCardSpec 

  enum class BlackjackChoices {
    STAY = 0, 
    HIT
  };

  using BlackjackPlayerActionSpec = spec::CategoricalArraySpec<BlackjackChoices, 2, 1>;
  using BlackjackActionSpec       = spec::CompositeArraySpec<BlackjackPlayerActionSpec>;

  struct BlackjackAction : environment::Action<BlackjackState, BlackjackActionSpec> {

    using environment::Action<BlackjackState, BlackjackActionSpec>::Action;

    template <std::size_t player>
    bool playerPass() const {
      return std::get<0>(*this).at(player) == 0 ;
    }

    template <std::size_t player> 
    void updateStatePlayer(StateType& newState) const {
      if(not this->playerPass<player>()){
        auto n = std::get<1>(newState.observable).at(player)++;
        std::get<0>(newState.observable).at(player, n ) = std::get<0>(newState.hidden).at(player, n );
      }
    }

    template <std::size_t ...player> 
    void updateState(StateType& newState, std::index_sequence<player...>) const {
      ( updateStatePlayer<player>(newState), ...);
    }


    StateType step(const StateType &state) const override { 
      // for each player being controlled and who has elected to 
      // hit a card we should reveal a hidden card.
      
      auto newState = state; 
      updateState(newState, std::make_index_sequence<players>());
      return newState; 
    }


  };


  using BlackjackStep = environment::Step<BlackjackAction>;


  struct BlackjackReward : reward::Reward<BlackjackAction> {

    static PrecisionType reward(const TransitionType &t) {
      auto scores = t.nextState.calculateHandValues();
      PrecisionType reward = 0.0;
      for (const auto& s: scores){
        if (s > 21)
          reward += 0.0F;
        reward += 21.0F - s;
      }
      return reward;
    }
  };

  using BlackjackReturn = returns::Return<BlackjackReward>;

  template <environment::RewardType REWARD_T, environment::ReturnType RETURN_T>
  struct BlackjackEnvironment : environment::Environment<BlackjackStep, REWARD_T, RETURN_T> {

    SETUP_TYPES_W_ENVIRON(
        SINGLE_ARG(environment::Environment<BlackjackStep, REWARD_T, RETURN_T>), 
        SINGLE_ARG(BlackjackEnvironment));

    StateType reset() override {

      const auto hidden =  ::policy::random_spec_gen<HiddenStateSpec>();
      auto observable =  ::policy::random_spec_gen<BlackjackStateSpec>();

      // It should be the case that the observale states 

      // Set observable card count to 2
      std::get<1>(observable)[0] = 2;
      std::get<1>(observable)[1] = 2;

      // Set the cards observable to the same as those hidden 
      std::get<0>(observable) = std::get<0>(hidden);
      // Hide all the cards which arent observable
      for (std::size_t i = 2; i<max_cards ; i++){
        std::get<0>(observable).at(0, i ) = 0;
        std::get<0>(observable).at(1, i ) = 0;
      }

      this->state = StateType(observable, hidden);
      return this->state;
    };


    template <std::size_t ...Is> 
    bool gameContinues_impl(TransitionType& t, std::index_sequence<Is...>){
     return (( (t.nextState.template calculateHandValue<Is>() < 21) and (not t.action.template playerPass<Is>())) | ... );
    }



    TransitionType step(const ActionSpace &action) override {

      auto t = TransitionType{this->state, action, StepType::step(this->state, action)};
      auto gameContinues = gameContinues_impl(t, std::make_index_sequence<players>{});

      if (not gameContinues)
        t.kind = environment::TransitionKind::TERMINAL;
      return t;
        
    }

    StateType getNullState() const override {
      return StateType();
    };

  };


}
