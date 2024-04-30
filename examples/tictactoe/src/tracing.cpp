#include "tracing.hpp"
#include "bindings.hpp"
#include <sstream>

namespace tictactoe {

DatabaseLogger::DatabaseLogger(std::shared_ptr<connection_t> &con) : connection(con) { connection->create_tables(); }

void Logger::log(const tictactoe::GameState &state, const tictactoe::Move &move) {
  currentTrajectory.trajectory.push_back({state, move});
}

void Logger::reset() { currentTrajectory.trajectory.clear(); }

void DatabaseLogger::save() const {
  if (not currentTrajectory.trajectory.empty()) {
    Episode episode;
    episode.id = -1; // -1 for auto increment
    episode.createdAt = time(0);
    episode.trajectory = bindings::to_json(currentTrajectory);
    episode.gameOver = currentTrajectory.trajectory.back().state.isGameOver();
    connection->insert_record(episode);
  }
}

} // namespace tictactoe