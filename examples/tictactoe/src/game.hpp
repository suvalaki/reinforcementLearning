#pragma once
#include <utility>
#include <vector>

namespace tictactoe {

enum class Player { EMPTY = 0, X, O };

struct Move {
  const Player player;
  const int row;
  const int column;
};

struct GameState {
  Player board[3][3] = {
      {Player::EMPTY, Player::EMPTY, Player::EMPTY},
      {Player::EMPTY, Player::EMPTY, Player::EMPTY},
      {Player::EMPTY, Player::EMPTY, Player::EMPTY}};
  Player currentPlayer = Player::X;
  Player winner = Player::EMPTY;
  bool playable = true;

  GameState() = default;
  void reset();
  void update(const Player &player, int row, int column);
  void update(const Move &move);
  bool checkWinner(const Player &player) const;
  bool isFull() const;
  bool isGameOver() const;
  Player getWinner() const;
};

struct StateAction {
  GameState state;
  Move action;
};

struct Trajectory {
  std::vector<StateAction> trajectory;
};

} // namespace tictactoe