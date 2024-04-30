#pragma once
#include <functional>
#include <optional>
#include <string>
#include <variant>

#include "game.hpp"
#include "tracing.hpp"

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

struct GameControl {

  GameState state = tictactoe::GameState();
  virtual std::string handleRequest(const std::string &json);
};

using game_control_factory_t = std::function<std::unique_ptr<GameControl>()>;

inline game_control_factory_t defaultGameControlFactory =
    std::function([]() { return std::make_unique<GameControl>(); });

struct LoggedGameControl : public GameControl {

  tictactoe::DatabaseLogger logger;

  LoggedGameControl(DatabaseLogger &logger) : logger(logger) {}
  virtual std::string handleRequest(const std::string &json) override;
};

} // namespace tictactoe::bindings
