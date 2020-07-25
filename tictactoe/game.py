import numpy as np
import random

class Board():

    def __init__(self, player_x_start=True, ):
        """TicTacToe Board and gamestate

        player_x_start: True when X goes first. False when O goes first
        """
        self.state = np.array([[None, None, None], [None, None, None], [None, None, None]]
        self.start_x = player_x_start
        if self.start_x:
            self.player_turn = 1 
        else:
            self.player_turn = 0
        self.turn = 0
        self.update_possible_next_state()


    def update_possible_next_state(self):
        self.pos_next_state = self.possible_next_state()


    def victory_condition(self):
        row_ret = self.row_victory()
        col_ret = self.col_victory()
        diag_ret = self.diag_ret() 

        if any([row_ret, col_ret, diag_ret]):
            return True 
        return False


    def row_victory(self):
        state = self.state
        for i in range(3):
            if (
                state[i, 0] == state[i, 1] 
                and state[i, 0] == state[i, 2] 
                and state[i, 0] is not None
            ):
                return True 
        return False


    def col_victory(self):
        state = self.state
        for i in range(3):
            if (
                state[0, i] == state[1, i] 
                and state[0, i] == state[2, i] 
                and state[0, i] is not None
            ):
                return True 
        return False


    def diag_victory(self):
        state = self.state
        if (
            state[0,0] == state[1,1] 
            and state[2, 2] == state[1, 1] 
            and state[1, 1] is not None
        ):
            return True 
        if (
            state[0, 2] == state[1, 1] 
            and state[2, 0] == state[1, 1] 
            and state[1, 1] is not None
        ):
            return True 
        return False

    def possible_next_state(self):
        possible_moves = [
            ((i, j), np.copy(self.state))
            for i in range(3)
            for j in range(3) 
            if self.state[i, j] is None
        ]
        
        # it is player O turn
        if self.player_turn == 0:
            for (i, j), stt for (i, j), stt in possible_moves:
                stt[i,j] = 0
            
        elif self.player_turn == 1:
            for (i, j), stt for (i, j), stt in possible_moves:
                stt[i,j] = 1
             
        return possible_next_state

    def __hash__(self):
        hsh = ""
        for i in range(2):
            for j in range(2):
                if self.state[i, j] == None:
                    hsh += "z"
                elif self.state[i, j] == 0:
                    hsh += "0"
                elif self.state[i, j] == 1:
                    hsh += "1"
        return hsh


class RandomOpponent():
    
    def __init__(self, board: Board, player_x=True):
        """This opponent creates a random strategy for each board 
        state to each next state
        """
        # Set which turn the player is allowed to make moves
        if board.player_x_start and player_x:
            self.play_even_turns = True 
        elif board.player_x_start and not player_x:
            self.play_even_turns = False
        elif not board.player_x_start and player_x:
            self.play_even_turns = False 
        else not board.player_x_start and not player_x:
            self.play_even_turns = True 
        self.player_x = player_x


    def create_random_policy(self, depth=0):
        if depth = 0:
            
        
