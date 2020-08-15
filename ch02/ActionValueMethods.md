## Associative Vs Non-Associative (Reinforcement Learning) Problems 

The <u>non-associative</u> setting is a simplification of the reinforcement learning problem where **most prior work involving evaluative feedback has already been done**. A non associative learner only learns to act on one setting. 

An evaluative learner uses training information that evaluates the performance of actions taken rather than indicating (instructing) which are the correct actions; **evaluative feedback is dependent on the action taken**. By contrast instructive feedback provides the correct action to take, no matter what our agent did; **instructive feedback is independent of the action taken**.

#### Evaluative Feedback Methods

- Value function estimation



## Exploration vs Exploitation

An agent can either exploit its current knowledge of the problem space or explore the situation. Exploitation does not allow the agent to modify its behaviour and so does not have the capability to improve the results the agent attains. Exploitation is the right thing to do to maximise the expected reward in a single timestep (or if the setting is played only once). Exploration allows the agent to modify its behaviour and so has the possibility to improve the return for games played in the future. There is constant conflict between exploiting current knowledge and exploring possible better actions. 

# Action-Value Methods

- Let $t\in T$ be time periods
- Let $a\in A$ be an action 
- Let $r_t(a)$ be the reward for taking action $a$ at time period $t$. In non-stationary settings it is possible for the reward of an action to change over time. 
- Let $q(a)=E(r(a))$ be the true value of action $a$. The true value of an action is the expected reward for taking action $a$. 
- Let $Q_t(a)=\widehat{q}(a)$  be the estimate of the true value of action $a$ after $t$ time periods
- Let $N_t(a)$ be the number of time periods action $a$ has been taken after $t$ time periods

## Multi Armed Bandits

The multi armed bandit problem is an example of a learning problem in a non-associative setting. 

- Let $t=0,...,T$ be discrete time periods an agent plays the game
- Let $\overrightarrow{X}=X_{1,t},...,X_{N,t}$ be $N$ random agents with unknown probability distributions at time period $T$. In the simplest setting $X_{i,t_1} \sim X_{i,t_2}$$\forall t_1, t_2 \in {T}$ (the distributions are the same over time). In more complex settings the distributions can shift.
- Let $a_t\in A$ be an action taken at time period $t$ from the set of all possible actions $A$. The possible actions at each time period being to choose a single index $1,...,N$. Each action has an expected reward.
- Let $v(t, a)=v(a_t)$  be the reward under action $a$ at time period $t$. The reward is the return from $a$.

At each time period we pick an index. We are seeking to consistently pick $X_{i,t}$ with the maximal value, so that at the end of the period $T$ we have accumulated the greatest possible reward. 

The multi-armed bandit problem illustrates the tension between 

- 