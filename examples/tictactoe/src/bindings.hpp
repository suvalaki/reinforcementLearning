#pragma once
#include "game.hpp"
#include <string>
#include <variant>

namespace tictactoe::bindings {

enum class VoidRequestType {
  RESET,
  CHECK_WINNER,
};

using Request = std::variant<Move, VoidRequestType>;

std::string to_json(const tictactoe::Player &player);
std::string to_json(const GameState &state);
std::string to_json(const tictactoe::Move &mv);
std::string to_json(const StateAction &trajectory);
std::string to_json(const Trajectory &trajectory);

tictactoe::Player from_json_player(const std::string &json);
tictactoe::Player from_json_Player(const std::string &json);
tictactoe::Move from_json_move(const std::string &json);

/** Assume json input looks like {"action": ACTION, "data": DATA} */
Request from_json_request(const std::string &json);
std::string update(GameState &state, const Request &request);
std::string checkWinner(const GameState &state, const Request &request);
std::string handleRequest(GameState &state, const std::string &json);

} // namespace tictactoe::bindings
