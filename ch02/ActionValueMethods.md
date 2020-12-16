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

## Aggregate updates
At any time preiod we need to update our estimate $Q_t(a)$. Instead of holding vectors of all values $r_t(a)$ we can perform itterative updates. 

### Stationary action values 
When $q(a)$ is invariant with time:
$$
\begin{aligned}
Q_{t+1}(a) &= \frac{1}{N_t(a)}\sum_{t\in N_t(a)} r_t(a) \\
&= \frac{1}{N_t(a)} \left( r_t(a) + \sum_{t\in N_{t-1}(a)} r_t(a) \right) \\
&= \frac{1}{N_t(a)} \left( r_t(a) + N_{t-1}(a)\left(\frac{1}{N_{t-1}(a)}\sum_{t\in N_{t-1}(a)} r_t(a) \right)\right) \\
&= \frac{1}{N_t(a)} \left( r_t(a) + N_{t-1}(a) * Q_{t-1}(a) \right) \\
&= \frac{1}{t} \left( r_t(a) + (t-1) * Q_{t-1}(a) \right) \\
&= Q_{t-1}(a) + \frac{1}{t} \left(  r_t(a) - Q_{t-1}(a) \right)\\ 
&= Q_{t-1}(a) +  \alpha_t * \left(  r_t(a) - Q_{t-1}(a) \right)
\end{aligned}
$$

As such the the update is a "sample weighted average". 

### Non-Stationairy action values
Often it is possible for action values to change with time. To account for this we need to weight more recent outcomes more than prior ones. 

We call $\alpha \in [0,1]$ the step size parameter. When $\alpha = 1/t$ we call this the simple moving average.  
his converges with probability 1 when :

$$
\sum_{t=1}^{\infty} \alpha_t(a) = \infty 
\hspace{5mm}
\text{and}
\hspace{5mm}
\sum_{t=1}^{\infty}\alpha_{t}^2(a) < \infty
$$
Non-covergence (which is the case for constant $\alpha_t$) is actually useful for non-stationairy problems because results <b>continue to vary in response to most recent rewards</b> (an never truly converge). If distribution is changing we want to be sensetive to these changes.

## Multi Armed Bandits

The multi armed bandit problem is an example of a learning problem in a non-associative setting. 

- Let $t=0,...,T$ be discrete time periods an agent plays the game
- Let $\overrightarrow{X}=X_{1,t},...,X_{N,t}$ be $N$ random agents with unknown probability distributions at time period $T$. In the simplest setting $X_{i,t_1} \sim X_{i,t_2}$$\forall t_1, t_2 \in {T}$ (the distributions are the same over time). In more complex settings the distributions can shift.
- Let $a_t\in A$ be an action taken at time period $t$ from the set of all possible actions $A$. The possible actions at each time period being to choose a single index $1,...,N$. Each action has an expected reward.
- Let $v(t, a)=v(a_t)$  be the reward under action $a$ at time period $t$. The reward is the return from $a$.

At each time period we pick an index. We are seeking to consistently pick $X_{i,t}$ with the maximal value, so that at the end of the period $T$ we have accumulated the greatest possible reward. 

The multi-armed bandit problem illustrates the tension between exploration and exploitation.

### Greedy Method
The greedy method is a purely expoloitative method; it does not explore in order to search for a better solution. The strategy is instantiated with an initial set of belief values for expected value of each choice. At every time step it selects the bandit it thinks will have the largest payoff, and then performs an update to its view of the the expected values.

The greedy action can only change when the estimated value for the greedy action dips bellow another non-greedy estimate. Hence low initial vonealues for value estimates will likely result in local non-optimal results. 

```
def exploit(banditValues):
    return argmax(banditValues)

def explore():
    pass 

def update(idx, value):
    # update the value of the bandit

def greedyMethod(bandits, iterations):
    for i in range(iterations):
        bandits.sample()
        idx = exploit(bandits.valueEstinmates)
        value = bandits.value(idx)
        update(idx, value)
    return bandits.valueEstimates

```
### Epsilon Greedy Method
Epslion greedy introduces a probabilistic search exploration step to the greedy method: at each time step with probability epsilon the strategy explores one of the non-greedy actions, otherwise the greedy action is selected. 


## Gradient Bandits

Rather than directly estimating the reward for a particular bandit selection or (or the probability that the bandit has the highest reward) we can instead learn some arbitrary "prefernce function"; a preference function can encode our relative selction.  We can then choose to select actions according to a softmax distribution. Hence:
$$
\text{prob of taking action a} = \pi_t(a)=\frac{\exp(H_t(A))}{\sum_{b=1}^{n}\exp(H_t(b))}
$$
To learn the preferences we:

- Start by setting all preferences to the same value 0 (no preference).
- 

 

