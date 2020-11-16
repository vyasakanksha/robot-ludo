from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import random
import numpy as np
import pandas as pd
from operator import add
import collections

class Agent(object):
    def __init__(self, params):
        self.turn = turn
        self.states = {}
        # The list of states, a linear representation of the 3x3 tic tac toe board
        self.state_order = []
        # The order in which the agent progressed through states to be able to
        # assign discounted rewards to older states.
        self.learning_rate = learning_rate
        self.decay = decay
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

    def set_state(self, old_board, action):
        """
        Store the action performed for a given state
        """
        state_key = Agent.serialize_board(old_board)
        self.state_order.append((state_key, action))
    
    def learn_by_temporal_difference(self, reward, new_state_key, state_key):
        """
        Implementation of the temporal difference formula.
        https://en.wikipedia.org/wiki/Temporal_difference_learning
        https://detailed.af/reinforcement/
        """
        old_state = self.states.get(state_key, np.zeros((3,3)))
        self.exploration_rate = max(self.exploration_rate - self.decay, 0.3)
        return self.learning_rate * ((reward * self.states[new_state_key]) - old_state)
    
    @staticmethod
    def serialize_state(board, turn):
        serialized_board = '{0:015b}'.format(board.counters - (board.redpen + sum(board.red))) + '{0:015b}'.format(board.counters - (board.bluepen + sum(board.blue))) + '{0:015b}'.format(board.redpen) + '{0:015b}'.format(board.bluepen) + str(turn)
        return serialized_board

    def on_reward(self, reward):
        """
        Assign rewards to actions performed on intermediate states.
        """
        if len(self.state_order) == 0:
            return None
        new_state_key, new_action = self.state_order.pop()
        # get the latest state and the action performed that led to the reward

        self.states[new_state_key] = np.zeros((3,3))
        # initialize the value with a zero matrix

        self.states[new_state_key].itemset(new_action, reward)
        # Assign the reward to this state

        while self.state_order:
            # while there is a stack of states (that were caused by actions performed)

            state_key, action = self.state_order.pop()
            # get the state and action performed on it

            reward *= self.discount_factor
            # Reduce the original reward (self.discount_factor is a number < 1)

            # Implementation of the value function
            if state_key in self.states:
                reward += self.learn_by_temporal_difference(reward, new_state_key, state_key).item(new_action)
                # If this state was encountered due to a different experiment, increase its previous value
                log('update learning', state_key, action, reward)
                log(self.states[state_key])
                self.states[state_key].itemset(action, reward)
            else:
                self.states[state_key] = np.zeros((3,3))
                reward = self.learn_by_temporal_difference(reward, new_state_key, state_key).item(new_action)
                self.states[state_key].itemset(action, reward)
                # If this state was not encountered before, assign it the discounted reward as its value
            new_state_key = state_key
            new_action = action

    def select_move(self, board):
        """
        Choose from exploration and exploitation.
        Epsilon greedy implementation for policy.
        http://home.deib.polimi.it/restelli/MyWebSite/pdf/rl5.pdf
        http://tokic.com/www/tokicm/publikationen/papers/AdaptiveEpsilonGreedyExploration.pdf
        """
        explore_message = 'Exploration turn'
        missing_experience_message = 'No experience for this state: explore'
        experience_present_message = 'Using previous experience'
        state_key = Agent.serialize_board(board)
        log('-' * 100)
        log('state key', state_key)
        p =  np.random.random()
        exploration = p < self.exploration_rate
        log(p, '<', self.exploration_rate)
        message = explore_message \
            if exploration \
            else missing_experience_message \
                if state_key not in self.states \
                else experience_present_message

        log(message)
        action = self.explore_board(board) \
                    if exploration or state_key not in self.states \
                    else self.exploit_board(state_key, board)
        log('Choose action', action)
        self.set_state(board, action)
        return action

    def explore_board(self, board, depth=0):
        """
        Find an empty cell from the board
        """
        moves = getmoves(board,turn,roll)
        mymove = random.choice(moves)
        pseudo_board = makemove(board,mymove,copyboard=True)
        state_key = Agent.serialize_board(pseudo_board)
        log(state_key)
        if state_key not in self.states or depth == 9:
            return mymove
        depth += 1
        return self.explore_board(board, depth=depth)

    def exploit_board(self, state_key, board):
        """
        Find the best action for the given state
        """
        state_values = self.states[state_key]
        # For the current state get the matrix of accumulated rewards
        log('State rewards', state_values)
        moves = getmoves(board,turn,roll)
        best_values = {}
        for idx, value in np.ndenumerate(state_values):
            if idx in moves:
                best_values[str(idx)] = value

        best_value_indices = [key
            for m in [max(best_values.values())]
                for key, val in best_values.items()
                    if val == m
        ]

        log('best_value_indices', best_value_indices)
        select_index = np.random.choice(len(best_value_indices))
        return ast.literal_eval(best_value_indices[select_index])


    def set_reward(self, player, crash):
        # get the latest state and the action performed that led to the reward
        if len(self.state_order) == 0:
            return None
        new_state_key, new_action = self.state_order.pop()

        # initialize the value with a zero matrix

        self.reward = 0
        if turn == 0:
            reward = (self.reward_temp(board.counters, board.redpen, board.red, board.safe)) - (self.reward_temp(board.counters, board.bluepen, board.blue, board.safe))
        if turn == 1:
            reward = (self.reward_temp(board.counters, board.bluepen, board.blue, board.safe)) - (self.reward_temp(board.counters, board.redpen, board.red, board.safe))
        return self.reward

    def reward_temp(counters, pen, inplay, safe):
        r = 0
        r += (counters - pen)*5 # 5 points for counters out of the pen
        r += (counters - (pen + sum(inplay))) * 20 # 20 points for counters which have made it home

        safe_list = list(filter(lambda x: x > 1, [a + b - 1 for a, b in zip(safe, inplay)]))
        r += sum(safe_list) * 10 # 10 points for counters which have made it home
        return r

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def replay_new(self, memory, batch_size):
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
        else:
            minibatch = memory
        for state, action, reward, next_state in minibatch:
            target = reward
            target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

    def train_short_memory(self, state, action, reward, next_state):
        target = reward
        target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, 11)))[0])
        target_f = self.model.predict(state.reshape((1, 11)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1, 11)), target_f, epochs=1, verbose=0)