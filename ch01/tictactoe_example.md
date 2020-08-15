## Motivating Reinforcement learning Tic-Tac-Toe
We can parameterise the problem of tic-tac-toe as a model free, finite state, discrete reinforcement learning problem with full future knowledge of states. We consider the current state of the board S_i and an action a_ij such that applicat5ion of a_ij to S_i produces board state S_j. Then there is a set of board states for the end of the game and the beginning. We can define a policy (a_1, a_2, ..., a_n) to be the set of actions mapping each state to the next. An optimal policy will have the highest probability of success. 

This problem has a number of attributes (and is simple in a number of ways)
- Finite Set: The complete set of states the board can be in is known and finite. This set of states is small. 
- Actions occur in discrete time steps and each player takes turns.
- The player is able to "look ahead" at all possible future states of the board and plan thier actions accordingly.

There are multiple approaches to this problem but we can exemplify the differences between reinforcement learning and other approaches by contrasting a value based approach against an evolutionairy hill climb. 



```{mermaid caption="the state"}
    stateDiagram-v2
    S0.0 --> S1.0 : action a00
    S0.0 --> S1.1 : action a01
    S0.0 --> S1.2 : action a02
    S1.0 --> S2.0 : action a10
    S1.0 --> S2.1 : action a11
```

### Evolutionary: Hill Climbing
One such evolutionairy method would approach the problem by proposing a policy and evaluating that policy performance over N games, then attempt incremental improvements via hill climbing.
```python 
def hill_climb(objective function f(x), x in R^d):
    while(improvements exist):
        adjust signle parameter x_i of x
        if f(x with change to x_i) > f(x without):
            accept change
        else:
            try another parameter
    return locally optimal solution

```
This methodology has a number of features:
- Holds the policy fixed to produce an unbiased estimate of winning under that policy. 
- Proposes a change to the policy only after a new unbiased estimate has been produced.
- The probability of winning the game is assigned to the entire policy. We cannot distinguish between different actions within the policy, and this cannot determine which actions are in themselves important. 
- Even given a large number of trails there may be states and actions which never occur so as above we cannot determine which actions in the policy were irrelevant to its success. 
- We do not model our opponent in any way. They are assumed to be fixed (we can get unbiased estimates of them).

### Value Based Method
For the game of tictactoe a greedy action will seek to take actions which have the highest probability of winning the game. A evolutionary method must create a policy (the total set of state-action pairs) the agent will take and then evaluate the effectiveness of the entire policy. In order to improve results that agent must modify the policy and re-evaluate effectiveness. When a modification improves the performance of the policy a greedy agent will accept it. However the performance evaluation is performed over the entire policy. It is not efficient (from a computational time standpoint) in evaluating how a given state-action pair would improve performance. To be more efficient in estimating the performance of taking any single action we can take a value based approach.

We can introduce a value function approach which better estimates the value/reward of taking a particular action (in this case having the board in a particular state). This iterative approach seeks to use "back-ups" to explore new action-state permutations to arrive at an optimal policy. Key to the value function is the idea that we can get the 'value' of a current state using the discount the value of future states by some step parameter `\alpha` to give us the value of the current state. 

Rather than having a fixed view of policy before we begin we seek at each state (in the discrete time game) the action which will yield the highest probability of victory. However the probability of victory, except in the case of states where the game is already over, unknown before we begin. Hence to instantiate our player we give probability 1 to all already victorious states, 0 to all already lost states and 0.5 to all other states. 

Then we play many games. During each game: each turn (given we are in state `S`) we lookup the table of probabilities of victory given we take action `a_S->S'` that moves the board into state `S'` and with probability `(1-t)` greedily select that future state which yields the best probability along the resultant state `S -action-> S'`. Sometimes we dont pick the best action, with probability `t` we explore an alternative possible future state via an exploratory move. 

To make our table of probabilities more accurate we update the value of each state after the conclusion of a game according to the value function: 

​	$V(s) \leftarrow v(s) + \alpha * [V(s') - v(s)]: \alpha \in  (0,1), s' = $future state after action a from s​

We begin by playing the game and setting a known action (strategy) for every state in the game. Without a loss of generality we assume to be playing as X and out opponent O. We further assume that we are playing against an opponent with a fixed policy too. We would like to always be in the position that we pick a strategy containing the states with the best possible probability of victory. This is easy for final states (where we have 3 X or 3 O in a row) because the game is already finished and so the probability of victory is deteministic either 1 or 0. 

```python
def tictactoe_value_backup(states S, reward v(s), exploration probability t):
    
    # initialise - given we are player X
    v(s) = 1 when s contains 3 Xs in a row
    v(s) = 0 when s contains 3 Os in a row
    v(s) = 0.5 otherwise 

    # itterate values
    for each game in GAMES:
        policy = []
        while game is not won:
            s = current_state(game)
            if players turn:
                explore = random var with prob t
                if not explore:
                    s* = next greedy state
                else:
                    s* = random choice of next states
                v(s) = v(s) + alpha * (v(s*) - v(s))
        continue playing until convergence is reached for values in v

    return v 
```
Then when playing the game thereafter we can choose simply not to perform future updates and exploration should we wish to fix the policy. However we can also update indefinitely and in-so-doing adjust to changes in the opponent if they arise. The moves taken converge to an optimal policy.   


### Evolutionary methods vs Value Based
Evolution holds a fixed policy for many games to produce an unbiased estimate of wins - Only after an estimate is produced is a greedy policy change proposed. But this is not needed in the case of reinforcement learning which was able to adapt to states as they arrived. And assigned probabilities to moving into particular board states themselves. 

### Further considerations
For generalised Reinforcement learing frameworks:

- There is no fixed requirement that we have a model of the world (and how our actions will modify the permissible state of the world). There exist models that do not have a model of the world (no environmental model).