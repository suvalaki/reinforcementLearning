import numpy as np
import random
from copy import deepcopy

SEED = 123
random.seed(SEED)


class Board():

    def __init__(self, player_x_start=True, ):
        """TicTacToe Board and gamestate

        player_x_start: True when X goes first. False when O goes first
        """
        self.state = np.array([[None, None, None], [None, None, None], [None, None, None]])
        self.start_x = player_x_start
        if self.start_x:
            self.player_turn = 1 
        else:
            self.player_turn = 0
        self.turn = 0
        self.game_over = False
        self.winner = None
        self.update_possible_next_state()


    def move(self, i,j):

        # depending on the player turn cadence we will have a different 
        # persons turn
        # we always see a new player run every even turn 

        assert(any([(i,j) == a for (a,b) in self.possible_next_state()]))
        assert(self.game_over != True)
        self.state[i,j] = self.player_turn
        self.update_possible_next_state()

        # check the game isnt finished this move
        if self.victory_condition(player=0):
            self.game_over = True
            self.winner = 0
            self.update_possible_next_state()
            return
        if self.victory_condition(player=1):
            self.game_over = True
            self.winner = 1
            self.update_possible_next_state()
            return
        if (self.state != None).sum() == 9:
            self.game_over = True 
            self.update_possible_next_state()
            return

        self.update_possible_next_state()

        # update which player is playing
        if self.player_turn == 0:
            self.player_turn  = 1
        else:
            self.player_turn = 0
        self.turn += 1 



    def print_state(self):
        pp = (
            f"\n{self.state[0,0]}|{self.state[1,0]}|{self.state[2,0]}"
            "\n-----\n"
            f"{self.state[0,1]}|{self.state[1,1]}|{self.state[2,1]}"
            "\n-----\n"
            f"{self.state[0,2]}|{self.state[1,2]}|{self.state[2,2]}\n"
        )
        print(pp)


    def update_possible_next_state(self):
        if self.game_over:
            self.pos_next_state = []
            return
        self.pos_next_state = self.possible_next_state()


    def victory_condition(self, player=0):
        row_ret = self.row_victory(player)
        col_ret = self.col_victory(player)
        diag_ret = self.diag_victory(player) 

        if any([row_ret, col_ret, diag_ret]):
            return True 
        return False


    def row_victory(self, player=0):
        state = self.state
        for i in range(3):
            if (
                player == state[i, 0]
                and state[i, 0] == state[i, 1] 
                and state[i, 0] == state[i, 2] 
                and state[i, 0] is not None
            ):
                return True 
        return False


    def col_victory(self, player=0):
        state = self.state
        for i in range(3):
            if (
                player == state[0,i]
                and state[0, i] == state[1, i] 
                and state[0, i] == state[2, i] 
                and state[0, i] is not None
            ):
                return True 
        return False


    def diag_victory(self, player=0):
        state = self.state
        if (
            player == state[0,0]
            and state[0,0] == state[1,1] 
            and state[2, 2] == state[1, 1] 
            and state[1, 1] is not None
        ):
            return True 
        if (
            player == state[0,2]
            and state[0, 2] == state[1, 1] 
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
            for (i, j), stt in possible_moves:
                stt[i,j] = 0
            
        elif self.player_turn == 1:
            for (i, j), stt in possible_moves:
                stt[i,j] = 1
             
        return possible_moves

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


    def state_as_tup(self):
        return (
            self.state[0,0], self.state[0,1], self.state[0,2],
            self.state[1,0], self.state[1,1], self.state[1,2],
            self.state[2,0], self.state[2,1], self.state[2,2],
        )


    def enumerate_all_future_states(self, depth=0):
        """Generate all legal position from subgames"""
        accumulator = [] 
        currstate = (deepcopy(self.state_as_tup()), deepcopy(self.player_turn))
        accumulator += [currstate]
        if self.game_over:
            return accumulator
        else:
            for k, ((i,j), stt) in enumerate(self.pos_next_state):
                cpy = deepcopy(self)
                cpy.move(i,j)
                cpy.update_possible_next_state()
                accumulator += cpy.enumerate_all_future_states(depth+1) 

            return list(set(accumulator))

    def enum_all_future_actions(self, depth=0):
        """move from state to next state"""
        accumulator = [] 
        currstate = (deepcopy(self.state_as_tup()), deepcopy(self.player_turn))
        if self.game_over:
            return accumulator
        else:
            for k, ((i,j), stt) in enumerate(self.pos_next_state):
                cpy = deepcopy(self)
                cpy.move(i,j)
                futurestate = (deepcopy(cpy.state_as_tup()), deepcopy(cpy.player_turn))
                accumulator += [(currstate, futurestate)]
                accumulator += cpy.enumerate_all_future_states(depth+1) 

            return list(set(accumulator))


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
        elif not board.player_x_start and not player_x:
            self.play_even_turns = True 
        self.player_x = player_x


    def create_random_policy(self, depth=0):
        if depth == 0:
            pass
            
        

if __name__ == "__main__":

    print("setting up board")
    brd = Board()
    print("getting all states")
    states = brd.enumerate_all_future_states()
    print(f"Finished getting all states: {len(states)}")
    print("getting all actions")
    actions = brd.enum_all_future_actions()
    print(f"Finished getting all actions: {len(actions)}")