#pragma once
#include <ctime>
#include <memory>

#include "game.hpp"
#include "zxorm/zxorm.hpp"

namespace tictactoe {

struct Episode {
  int id;
  int createdAt;
  std::string trajectory; // Trajectory
  bool gameOver;
};

using episode_t = zxorm::Table<
    "epsiode",
    Episode,
    zxorm::Column<"id", &Episode::id, zxorm::PrimaryKey<zxorm::conflict_t::abort>>,
    zxorm::Column<"createdAt", &Episode::createdAt, zxorm::NotNull<>, zxorm::Unique<>>,
    zxorm::Column<"trajectory", &Episode::trajectory, zxorm::NotNull<>, zxorm::Unique<>>>;

using connection_t = zxorm::Connection<episode_t>;

struct Logger {
  Trajectory currentTrajectory;

  void log(const tictactoe::GameState &state, const tictactoe::Move &move);
  void reset();
  virtual void save() const = 0;
};

struct DatabaseLogger : Logger {
  std::shared_ptr<connection_t> connection;

  DatabaseLogger(std::shared_ptr<connection_t> &con);
  void save() const override;
};

} // namespace tictactoe
