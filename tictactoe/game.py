import numpy as np
import random
from copy import deepcopy
import logging
import sys
import abc
from typing import List, Tuple, Optional

SEED = 123
random.seed(SEED)

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


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
        self.all_states = None 
        self.all_actions = None


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
        currstate = (self.state_as_tup(), self.player_turn)
        accumulator += [currstate]
        if self.game_over:
            return accumulator
        else:
            for k, ((i,j), stt) in enumerate(self.pos_next_state):
                cpy = deepcopy(self)
                if cpy.all_states != None: 
                    cpy.all_states = None
                if cpy.all_actions != None: 
                    cpy.all_actions = None
                cpy.move(i,j)
                cpy.update_possible_next_state()
                accumulator += cpy.enumerate_all_future_states(depth+1) 

            return list(set(accumulator))


    def enum_all_future_actions(self, depth=0):
        """move from state to next state"""
        accumulator = [] 
        currstate = (self.state_as_tup(), self.player_turn)
        if self.game_over:
            return accumulator
        else:
            for k, ((i,j), stt) in enumerate(self.pos_next_state):
                cpy = deepcopy(self)
                if cpy.all_states != None: 
                    cpy.all_states = None
                if cpy.all_actions != None: 
                    cpy.all_actions = None
                cpy.move(i,j)
                futurestate = (cpy.state_as_tup(), cpy.player_turn)
                accumulator += [(currstate, futurestate)]
                accumulator += cpy.enum_all_future_actions(depth+1) 

            return list(set(accumulator))

    def set_state_space(self):
        logger.debug("Enumerating all board states")
        self.all_states = self.enumerate_all_future_states() 

    def set_action_space(self):
        logger.debug("Enumerating all possible actions")
        self.all_actions = self.enum_all_future_actions() 
    
    def set_space(self):
        self.set_state_space()
        self.set_action_space()



class Player(abc.ABC):
    def __init__(self, name:str, board:Board, player_x:bool = True, policy = None):
        self.name = name 
        self.player_x = player_x
        self.board = board
        self._policy = policy
        self.set_turn_order()


    @property
    @abc.abstractmethod
    def policy(self):
        return self._policy 

    @policy.setter
    @abc.abstractmethod 
    def policy(self, p):
        if policy == None:
            self._policy = None 
        else:
            self._policy = p

    @abc.abstractmethod
    def get_move_given_board(self, brd: Board) -> Tuple[int, int]:
        return (0,0)

    def set_turn_order(self):
        """Set which turn we are playing."""
        # Set which turn the player is allowed to make moves
        if self.board.start_x and self.player_x:
            self.play_even_turns = True 
        elif self.board.start_x and not self.player_x:
            self.play_even_turns = False
        elif not self.board.start_x and self.player_x:
            self.play_even_turns = False 
        elif not self.board.start_x and not player_x:
            self.play_even_turns = True 


    def get_move(self, state0, state1):

        loc_diff = [state0[0][i] != state1[0][i] for i in range(9)]
        loc_map = {
            0: (0,0), 1:(0,1), 2: (0,2),
            3: (1,0), 4:(1,1), 5: (1,2),
            6: (2,0), 7:(2,1), 8: (2,2),
        }
        location = [i for i,x in enumerate(loc_diff) if x][0]
        return loc_map[location]



class DeterministicPolicyRangomOpponent(Player):
    
    def __init__(self, name:str, board: Board, player_x=True):
        """This opponent creates a random strategy for each board 
        state to each next state
        """
        logger.debug(f"Creating Random Oppoenent: {name}")
        super().__init__(name, board, player_x)
        self.create_random_policy()
        logger.debug(f"Created {name}")

    @property
    def policy(self):
        return self._policy

    @policy.setter 
    def policy(self, p):
        self._policy = p
        self._policy_dict = {k:v for (k,v) in p}


    def get_move_given_board(self, brd: Board):

        curr_state = (brd.state_as_tup(), brd.player_turn)
        futr_state = self._policy_dict[curr_state]
        move = self.get_move(curr_state, futr_state)
        return move


    def create_random_policy(self):

        logger.debug(f"Creating random policy for {self.name}")
        if self.board.all_states == None:
            self.board.set_state_space()
        all_states = self.board.all_states
        
        if self.board.all_actions == None:
            self.board.set_action_space()
        all_actions = self.board.all_actions
        
        # for each current state pick a single future state as the action

        logger.debug("Setting action space")
        policy = []
        for state in all_states:
            possible_actions = list(filter(lambda x: x[0] == state, all_actions))
            # pick a random action 
            if len(possible_actions) > 0:
                policy.append((state, random.choice(possible_actions)[1]))
        logger.debug("Finished setting action space")

        self.policy = policy
        self.policy_dict = {k:v for (k,v) in policy}
            



class Game():

    def __init__(self, board:Board, player_0: Player, player_1: Player):

        self.board = board 
        self.player_0 = player_0 
        self.player_1 = player_1 
        self.moves = []


    def run_policy(self):

        logger.info("Starting Game")
        player_turn = 0
        while(self.board.game_over == False):
            # play a turn 
            if player_turn == 0:
                move = self.player_0.get_move_given_board(self.board)
                self.board.move(*move)
                player_turn = 1
                self.moves.append((move, 0))
                logger.info(f"Player 0 moved: {move}")
            elif player_turn == 1:
                move = self.player_1.get_move_given_board(self.board)
                self.board.move(*move)
                player_turn = 0
                self.moves.append((move, 1))
                logger.info(f"Player 1 moved: {move}")
            
        winner = (
            "None" 
            if self.board.winner == None 
            else [self.player_0.name, self.player_1.name][self.board.winner]
        )

        logger.info(f"Game Finished. {winner}")




        

if __name__ == "__main__":

    brd = Board()
    brd.set_space()
    player0 = DeterministicPolicyRangomOpponent("player_0", brd)
    player1 = DeterministicPolicyRangomOpponent("player_1", brd, False)

    game = Game(brd, player0, player1)
    game.run_policy()
    brd.print_state()
