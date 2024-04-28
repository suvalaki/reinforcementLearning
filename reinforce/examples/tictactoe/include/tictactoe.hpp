#pragma once
#include <array>

#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

#include "environment.hpp"
#include "policy/random_policy.hpp"

namespace examples::tictactoe {

/** In tic-tac-toe the state and the moves you can make correspond to the locaiton on the board.
 * Player positions can be considered to take up these squares. As such a vector of squares for
 * each player can be used to maintain the board state. When niether player is using a square
 * each players vector will be zero. 
 * 
 * player0: [0, 0, 1, 0, 1, 0]
 * player2: [1, 0, 0, 1, 0, 1]
 * free:    [0, 1, 0, 0, 0, 0] . (only 1 free space left)
 * 
 * The game can also expand past 3x3. You could make a tic-tac-toe of any dimension.
*/

enum class Players {
    PLAYER0 = 0,
    PLAYER1,
};

enum class Squares {
    TOP_LEFT = 0,
    TOP_MID,
    TOP_RIGHT,
    MID_LEFT,
    MID_MID,
    MID_RIGHT,
    BOTTOM_LEFT,
    BOTTHOM_MID,
    BOTTOM_RIGHT,
};

constexpr nSquares = std::numeric_limits<typename std::underlying_type<Squares>::type>::max();

using PlayerStateSpec = spec::BoundedAarraySpec<int, 0, 1, nSquares>;
using PlayerActionSpec = spec::CategoricalArraySpec<Squares, nSquares, 1>;

using GameStateSpec = spec::CompositeArraySpec<PlayerSpec, PlayerSpec>;
using GameActionSpec = spec::CompositeArraySpec<PlayerActionSpec, PlayerActionSpec>;

using rewardPrecision = float;
using GameState = environment::State<rewardPrecision, GameStateSpec>;

using PlayerAction = environment::Action<GameState, GameActionSpec> {
    using environment::Action<GameState, GameActionSpec>::Action;
    Squares playerAction() const { return Squares{std::get<0>(*this).at(0)}; }
}


using PlayerStep = environment::Step<PlayerAction>;

struct PlayerReward : reward::Reward::PlayerAction> {


    static bool consequtiveTrue(const typename PlayerStateSpec::DataType &playerState, const Squares &s0, const Squares &s1, const Squares &s2){
        return playerState.at(s0) == 1 && playerState.at(s1) == 1 && playerState.at(s2) == 1;
    }


    static bool isWin(const typename PlayerStateSpec::DataType &playerState){
        // Check if the player has won

        // Check rows
        if (consequtiveTrue(playerState, Squares::TOP_LEFT, Squares::TOP_MID, Squares::TOP_RIGHT)){
            return true;
        }
        if (consequtiveTrue(playerState, Squares::MID_LEFT, Squares::MID_MID, Squares::MID_RIGHT)){
            return true;
        }
        if (consequtiveTrue(playerState, Squares::BOTTOM_LEFT, Squares::BOTTOM_MID, Squares::BOTTOM_RIGHT)){
            return true;
        }

        // Check columns
        if (consequtiveTrue(playerState, Squares::TOP_LEFT, Squares::MID_LEFT, Squares::BOTTOM_LEFT)){
            return true;
        }
        if (consequtiveTrue(playerState, Squares::TOP_MID, Squares::MID_MID, Squares::BOTTOM_MID)){
            return true;
        }
        if (consequtiveTrue(playerState, Squares::TOP_RIGHT, Squares::MID_RIGHT, Squares::BOTTOM_RIGHT)){
            return true;
        }

        // Check diagonals
        if (consequtiveTrue(playerState, Squares::TOP_LEFT, Squares::MID_MID, Squares::BOTTOM_RIGHT)){
            return true;
        }
        if (consequtiveTrue(playerState, Squares::TOP_RIGHT, Squares::MID_MID, Squares::BOTTOM_LEFT)){
            return true;
        }

        return false;

    }

    static bool boardIsFull(const typename GameStateSpec::DataType &gameState){
        // Check if the board is full - By looking at both players states
        for (int i = 0; i < nSquares; i++){
            if (gameState.at(Players::PLAYER0).at(i) == 0 || gameState.at(Players::PLAYER1).at(i) == 0){
                return false;
            }
        }
        return true;
    }

    // Assuming starting player
    template <int PLAYER>
    static PrecisionType player_reward(const TransitionType &t){

        if (isWin(std::get<PLAYER>(t.nextState.observable))){
            return 1.0F;
        }
        if (isWin(std::get<1-PLAYER>(t.nextState.observable))){
            return -1.0F;
        }
        if (boardIsFull(std::get<0>(t.nextState.observable))){
            return 0.5F;
        }
        return 0.0F;

    }

};

struct Player0Reward : PlayerReward<Players::PLAYER0> {
    static PrecisionType reward(const TransitionType &t){
        return player_reward<Players::PLAYER0>(t);
    }
};

struct Player1Reward : PlayerReward<Players::PLAYER0> {
    static PrecisionType reward(const TransitionType &t){
        return player_reward<Players::PLAYER0>(t);
    }
};

using Player0Return = reward::Return<Player0Reward>;
using Player1Return = reward::Return<Player1Reward>;


template <environment::RewardType REWARD_T, environment::ReturnType RETURN_T>
using TicTacToeEnvironment = environment::FiniteEnvironment<GameState, REWARD_T, RETURN_T>{
    SETUP_TYPES_W_ENVIRON(
        SINGLE_ARG(environment::Environment<GameState, REWARD_T, RETURN_T>), SINGLE_ARG(TicTacToeEnvironment));


  StateType reset() override {
    return GameState{PlayerStateSpec::DataType{0}, PlayerStateSpec::DataType{0}};
  };


  std::unordered_set<StateType, typename StateType::Hash> getAllPossibleStates() const override {
    auto states = std::unordered_set<StateType, typename StateType::Hash>{};

    // All possible allocations of player 0 and player 1 to the board
    // need to handle recursively as the game progrsses through a number of turns
    // and the board fills up. So we should create a tree of possible states
    // and then traverse it to get all possible states.

    const auto startingGameState = GameState{PlayerStateSpec::DataType{0}, PlayerStateSpec::DataType{0}};
    const auto player0 = PlayerStateSpec::DataType{0};
    const auto player1 = PlayerStateSpec::DataType{0};

    // Recursive function to generate all possible states
    auto generateStates = [&](const auto &gameState, const auto &player0Move, const auto &player1Move) {

        auto newGameState = std::copy(gameState);
        newGameState.at(Players::PLAYER0).at(player0Move) = 1;
        newGameState.at(Players::PLAYER1).at(player1Move) = 1;

        auto gameOver = PlayerReward::isWin(newGameState.at(Players::PLAYER0)) 
            || PlayerReward::isWin(newGameState.at(Players::PLAYER0))
            || PlayerReward::boardIsFull(newGameState);
        if (gameOver){
            states.insert(GameState{player0Move, player1Move});
            return;
        }

        const auto nUsedSquaresPlayer0 = std::accumulate(newGameState.at(Players::PLAYER0).begin(), newGameState.at(Players::PLAYER0).end(), 0, std::plus<int>());
        const auto nUsedSquaresPlayer1 = std::accumulate(newGameState.at(Players::PLAYER1).begin(), newGameState.at(Players::PLAYER1).end(), 0, std::plus<int>());
        const auto nUsedSquare = nUsedSquaresPlayer0 + nUsedSquaresPlayer1;

        // Generate all possible states for the next turn
        // player can make moves in empty squares

        for (int i = 0; i < nSquares; i++){

            const auto allowedMove = gameState.at(Players::PLAYER0).at(i) == 0 and gameState.at(Players::PLAYER1).at(i) == 0;
            if (allowedMove){
                auto nextGameState = std::copy(gameState);

                for (int j = 0; j < nSquares; j++){
                    const auto allowedMove = gameState.at(Players::PLAYER0).at(j) == 0 and gameState.at(Players::PLAYER1).at(j) == 0;
                }

            }


        }


    };



    return states;
  };

};



} // namespace examples::tictactoe
