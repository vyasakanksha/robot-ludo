# Multi-player game playing agent with Reinforcement Learning (Ludo)

## The Goal

The goal of this project was to create a bot/agent to play Ludo, as a gentle introduction to Reinforcement Learning. But what it really turned out to be was a 
jouney through the wild world of GitHub to learn about a completely new topic of which I had minimal prior knowledge. Navigating through the different projects gave 
me as a more general study of Reinforcement Learning to create game playing agents and the challenge of tweaking it to this specific problem. The path I took 
is detailed here. I should note, that this project is still WIP, so if you don't like anti-climactic endings then don't read on. But, if someone can help
me understand what I am doing wrong, I would love some help!

## Ludo 
Many summer afternoons have been spent on this game. A details explination about the games and the rules of this paricular challenge are detailed here.

### Core idea behind Reinforcement Learning
Modeled as a Markov Decision Process the value function attempts to maximizes the return by maximizing the cummative reward at the end of the game (final state)
from the current state.

## Attempt 1 - Deep Q-Learning algorithm for Snake

A Deep Reinforcement learning Algorithm using Q-tables is explained and implemented here as an agent to play snake. It is able to create a pretty good bot in about 
15 mins to traning (though sometimes it gets caught in a loop). However, I struggled with converting the single-player-game into a multi-player game. 

### Core idea behind Q-tables
Store the maximum furure reward for every action from every state.

$Q^{pi}(s_t, a_t) = \leftarrow \underbrace {Q(s_{t},a_{t})} _{\text{old value}}+\underbrace {\alpha } _{\text{learning rate}}\cdot \overbrace {{\bigg (}\underbrace {\underbrace {r_{t}} _{\text{reward}}+\underbrace {\gamma } _{\text{discount factor}}\cdot \underbrace {\max _{a}Q(s_{t+1},a)} _{\text{estimate of optimal future value}}} _{\text{new value (temporal difference target)}}-\underbrace {Q(s_{t},a_{t})} _{\text{old value}}{\bigg )}} ^{\text{temporal difference}}}
$

### References

https://towardsdatascience.com/snake-played-by-a-deep-reinforcement-learning-agent-53f2c4331d36

https://github.com/maurock/snake-ga

## Attempt 2 - Temporal Difference algorithm for Tic Tac Toe

The TD algorithm implemented to train a bot to play tic-tac-toe. The challenge here was that tic-tac-toe has a very small gamespace, and the algorithm was able to 
map every possible state the of the board and make decisions accordingly. This is not possible with ludo. 

### References
https://medium.com/vernacular-ai/reinforcement-learning-step-by-step-17cde7dbc56c

https://github.com/ltbringer/tic_tac_toe/blob/master/agent.py

## Interesting Observation (sun)
There is a famous impletentation of the Temporal Difference algorithm (TD-Lambda) to build a backgammon-playing agent know as TDGammon (circa 1992), a huge
advancement for its time. As it happens, Ludo is just a less complicated version of backgammon. 

## Attempt 3 - TDGammon
This is the approach with which I got far enough to experiment. I was able to successfully implement a verison of ludo which will fit with the TDgammon model. Then
I experimented with features to make it 

### References
https://medium.com/jim-fleming/before-alphago-there-was-td-gammon-13deff866197

https://github.com/fomorians/td-gammon

## Attempt 4 - Deep Q-Learning for Connect4 (TO DO)

https://towardsdatascience.com/playing-connect-4-with-deep-q-learning-76271ed663ca

