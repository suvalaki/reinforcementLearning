#include "bindings.hpp"
#include <sstream>
#include <stdexcept>

std::string tictactoe::bindings::to_json(const tictactoe::Player &Player) {
  switch (Player) {
  case Player::EMPTY:
    return "null";
  case Player::X:
    return "\"X\"";
  case Player::O:
    return "\"O\"";
  }
}

std::string tictactoe::bindings::to_json(const GameState &state) {
  /** Example Json format: {"board":[[null,"X",null],[null,null,"O"],[null,null,null]],"currentPlayer":1} */
  std::stringstream result;
  result << "{";
  result << "\"board\":[";
  for (int i = 0; i < 3; i++) {
    result << "[";
    for (int j = 0; j < 3; j++) {
      result << to_json(state.board[i][j]);
      if (j < 2) {
        result << ",";
      }
    }
    result << "]";
    if (i < 2) {
      result << ",";
    }
  }
  result << "],";
  result << "\"currentPlayer\":";
  result << to_json(state.currentPlayer);
  result << ",\"winner\":";
  result << to_json(state.winner);
  result << ",\"playable\":";
  result << (state.playable ? "true" : "false");
  result << "}";
  return result.str();
}

tictactoe::Player tictactoe::bindings::from_json_player(const std::string &json) {
  if (json == "X") {
    return Player::X;
  } else if (json == "O") {
    return Player::O;
  }
  return Player::EMPTY;
}

tictactoe::Move tictactoe::bindings::from_json_move(const std::string &json) {
  // Example json format: {"player": "X", "row": 0, "column": 0}
  auto player_str = json.substr(json.find("player") + 9, 1);
  auto x = json.substr(json.find("row") + 5, 1);
  auto y = json.substr(json.find("col") + 5, 1);
  auto player = from_json_player(player_str);
  auto row = std::stoi(x);
  auto column = std::stoi(y);
  return tictactoe::Move{player, row, column};
}

tictactoe::bindings::Request tictactoe::bindings::from_json_request(const std::string &json) {

  static const std::string action_str = "\"action\":\"";
  auto idx0 = json.find(action_str) + action_str.size();
  auto idx1 = json.substr(idx0).find("\"");

  std::string action = json.substr(idx0, idx1);
  if (action == "move") { // {"action": "move", "data":[]}
    std::string data = json.substr(json.find("data") + 6);
    data = data.substr(0, data.find("}"));
    return from_json_move(data);
  } else if (action == "reset") { // {"action": "reset"}
    return VoidRequestType::RESET;
  } else if (action == "check") { // {"action": "check"}
    return VoidRequestType::CHECK_WINNER;
  } else {
    throw std::invalid_argument("Invalid action: \"" + action + "\" payload: \"" + json + "\"");
  }
}

std::string tictactoe::bindings::update(GameState &state, const Request &request) {
  // returns a valid json string of the current game state
  if (std::holds_alternative<Move>(request)) {
    Move move = std::get<Move>(request);
    state.update(move);
    return to_json(state);
  } else {
    throw std::invalid_argument("Invalid request");
  }
}

std::string tictactoe::bindings::checkWinner(const GameState &state, const Request &request) {
  // Returns a valid json string of the format {"winner": WINNER}
  if (std::holds_alternative<VoidRequestType>(request)) {
    const auto winner = state.getWinner();
    return "{\"winner\": " + to_json(winner) + "}";
  } else {
    throw std::invalid_argument("Invalid request");
  }
}

std::string tictactoe::bindings::handleRequest(GameState &state, const std::string &json) {
  Request request = from_json_request(json);
  if (std::holds_alternative<Move>(request)) {
    return update(state, request);
  } else if (std::holds_alternative<VoidRequestType>(request)) {
    auto parsedRequest = std::get<VoidRequestType>(request);
    switch (parsedRequest) {
    case VoidRequestType::RESET:
      state.reset();
      return to_json(state);
    case VoidRequestType::CHECK_WINNER:
      return checkWinner(state, request);
    default:
      throw std::invalid_argument("Invalid request");
    }
  }
  return "";
}