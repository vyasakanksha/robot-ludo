# Multi-player game playing agent with Reinforcement Learning (Ludo)

## The Goal

The goal of this project was to create a bot/agent to play Ludo, as a gentle introduction to Reinforcement Learning. 

What it has actually turned into, is a journey through the wild world of GitHub as an attempt to learn about a completely new topic of which I had minimal prior knowledge. As I navigated through different projects which demonstated implementation (along with some theory) for varied algorithms to create game playing agents, it provided a more general study of Reinforcement Learning and a few different starting points for solving this problem. The path I took is detailed here. 

I should note that this project is still WIP, so if you don't like anticlimactic endings then don't read on. But, if someone can help me understand what I am doing wrong, I would love some help!

## Ludo 
Many summer afternoons have been spent on this game. [A detailed explanation about the games and the rules of this particular challenge are detailed here.](https://github.com/vyasakanksha/robot-ludo/blob/main/Ludo.md)


## Attempt 1 - The Naive Bot

I started with naive bot which considered the following: 
1. The number of pieces which were back home
2. The number of pieces out of the pen
3. The number of pieces on safe squares 

And gave them respective weights. This agent beat the simple random agent 90% of the team. The whole game now is to optimize these weights. Can reinforcement learning help us with that?

The `model.py` file has been modified to add `simple_player4` and `simple_payer5`, which basically do the same thing. I was playing them against each other to experiment with the weights for each feature. The final agent function can be found in `akanksha_ludo.py`.

### Core idea behind Reinforcement Learning
Modeled as a Markov Decision Process the value function attempts to maximizes the return by maximizing the commutative reward at the end of the game (final state)
from the current state.

## Attempt 2 - Deep Q-Learning algorithm for Snake

A Deep Reinforcement learning Algorithm using Q-tables is explained and implemented here as an agent to play snake. It is able to create a pretty good bot in about 
15 mins to training (though sometimes it gets caught in a loop). However, I struggled with converting the single-player-game into a multiplayer game. 

The whole `snakega` repo has been cloned in the `snake-ga` folder and `train_from_snake.py` is a very incomplete attempt converting the model to play ludo. 

### Core idea behind Q-tables
Store the maximum future reward for every action from every state.

### References

https://towardsdatascience.com/snake-played-by-a-deep-reinforcement-learning-agent-53f2c4331d36

https://github.com/maurock/snake-ga

## Attempt 3 - Temporal Difference algorithm for Tic Tac Toe

The TD algorithm was implemented to train a bot to play tic-tac-toe. The challenge here was that tic-tac-toe has a very small gamespace, and the algorithm was able to map every possible state of the board and make decisions accordingly. This is not possible with ludo. 

The whole `tic-tac-toe` repo has been cloned in the `tic-tac-toe` folder and `agent_from_tictactoe.py` is a very incomplete attempt at using this algorithm to create a Ludo playing agent.

### Core idea behind TD
Sample the environment to adjust the model even before the final outcome is known, and estimate the value function.

![\begin{align*}
V^{\pi}(s) = E_{a ~ \pi}\big\{\sum^{\infinity}_{t=0}\gamma^t r_t (a_t) \big \vert s_0 = s\big\} 
\; \text{where:} \\ \\ \\
s \leftarrow \text{States} \\
r \leftarrow \text{Reward} \\
\gamma \leftarrow \text{decay factor} \\
V \leftarrow \text{State value function of MDP} \\
\end{align*}
](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0AV%5E%7B%5Cpi%7D%28s%29+%3D+E_%7Ba+%7E+%5Cpi%7D%5Cbig%5C%7B%5Csum%5E%7B%5Cinfinity%7D_%7Bt%3D0%7D%5Cgamma%5Et+r_t+%28a_t%29+%5Cbig+%5Cvert+s_0+%3D+s%5Cbig%5C%7D+%0A%5C%3B+%5Ctext%7Bwhere%3A%7D+%5C%5C+%5C%5C+%5C%5C%0As+%5Cleftarrow+%5Ctext%7BStates%7D+%5C%5C%0Ar+%5Cleftarrow+%5Ctext%7BReward%7D+%5C%5C%0A%5Cgamma+%5Cleftarrow+%5Ctext%7Bdecay+factor%7D+%5C%5C%0AV+%5Cleftarrow+%5Ctext%7BState+value+function+of+MDP%7D+%5C%5C%0A%5Cend%7Balign%2A%7D%0A)

From this we get: 

![\begin{align*}
V^{\pi}(s) = E_{\pi}\big\{r_0 + \gamma V^{\pi}(s_1) \big \vert s_0 = s \big\}
\end{align*}](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0AV%5E%7B%5Cpi%7D%28s%29+%3D+E_%7B%5Cpi%7D%5Cbig%5C%7Br_0+%2B+%5Cgamma+V%5E%7B%5Cpi%7D%28s_1%29+%5Cbig+%5Cvert+s_0+%3D+s+%5Cbig%5C%7D%0A%5Cend%7Balign%2A%7D)

Fill the table ![\begin{align*}
V(s)
\end{align*}
](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+%5Cbegin%7Balign%2A%7D%0AV%28s%29%0A%5Cend%7Balign%2A%7D%0A)
 with a positive learning rate ![\begin{align*}
\alpha
\end{align*}
](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+%5Cbegin%7Balign%2A%7D%0A%5Calpha%0A%5Cend%7Balign%2A%7D%0A)

![\begin{align*}
V(s) \leftarrow V(s) = \alpha (r + \gamma V(s') - V(s))
\end{align*}
](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+%5Cbegin%7Balign%2A%7D%0AV%28s%29+%5Cleftarrow+V%28s%29+%3D+%5Calpha+%28r+%2B+%5Cgamma+V%28s%27%29+-+V%28s%29%29%0A%5Cend%7Balign%2A%7D%0A)


### References
https://medium.com/vernacular-ai/reinforcement-learning-step-by-step-17cde7dbc56c

https://github.com/ltbringer/tic_tac_toe/blob/master/agent.py

## Interesting Observation (sun)
There is a famous implementation of the Temporal Difference algorithm (TD-Lambda) to build a backgammon-playing agent known as TDGammon (circa 1992), a huge
advancement for its time. As it happens, Ludo is just a less complicated version of backgammon. 

## Attempt 4 - TDGammon
This showed some promise. With this approach far enough to experiment. I was able to successfully implement a version of ludo which will fit with the TDgammon model. I then added the same features I had added to my smartest naive bot, and experimented with them. The goal is to get this model to learn the respective weights for those features so it can repeatedly beat my naive bot. Unfortunately, our agent will not get smarter with more training.

All the code is on the `ludo` folder, and it can also be tested with `play.py`

The next step would be to try and encode the whole board as features. 

### References
https://medium.com/jim-fleming/before-alphago-there-was-td-gammon-13deff866197

https://github.com/fomorians/td-gammon

## Attempt 4 - Deep Q-Learning for Connect4 (TO DO)

https://towardsdatascience.com/playing-connect-4-with-deep-q-learning-76271ed663ca

