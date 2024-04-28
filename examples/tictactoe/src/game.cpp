#include "game.hpp"

using namespace tictactoe;



void GameState::reset() {
    currentPlayer = Player::X;
    winner = Player::EMPTY;
    playable = true;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            board[i][j] = Player::EMPTY;
        }
    }
}

void GameState::update(const Player& player, int row, int column) {
    if (board[row][column] == Player::EMPTY and currentPlayer == player) {
        board[row][column] = player;
        currentPlayer = (player == Player::X) ? Player::O : Player::X;
    }
}
void GameState::update(const Move& move) {
    update(move.player, move.row, move.column);
    winner = getWinner();
    playable = !isGameOver();
}

bool GameState::checkWinner(const Player& player) const {
    // Check rows
    for (int i = 0; i < 3; i++) {
        if (board[i][0] == player && board[i][1] == player && board[i][2] == player) {
            return true;
        }
    }

    // Check columns
    for (int i = 0; i < 3; i++) {
        if (board[0][i] == player && board[1][i] == player && board[2][i] == player) {
            return true;
        }
    }

    // Check diagonals
    if (board[0][0] == player && board[1][1] == player && board[2][2] == player) {
        return true;
    }
    if (board[0][2] == player && board[1][1] == player && board[2][0] == player) {
        return true;
    }

    return false;
}

bool GameState::isFull() const {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (board[i][j] == Player::EMPTY) {
                return false;
            }
        }
    }
    return true;
}

bool GameState::isGameOver() const {
    return checkWinner(Player::X) || checkWinner(Player::O) || isFull();
}

Player GameState::getWinner() const {
    if (checkWinner(Player::X)) {
        return Player::X;
    } else if (checkWinner(Player::O)) {
        return Player::O;
    } else {
        return Player::EMPTY;
    }
}
