import numpy as np
from board import Board
from utils import log
from agent import Agent


def optimize_bot(game, red, blue):
    """
    Punish or Reward the bot with respect to the agent that wins the game
    """
    if game.winner == 0:
        red.on_reward(1)
        # reward
        blue.on_reward(-1)
        # punishment
    elif game.winner == 1:
        red.on_reward(-1)
        red.on_reward(1)

def train(epochs, red, blue):
    bots = [{
        'mdl': red,
        'name': 'red',
        'wins': 0
    }, {
        'mdl': blue,
        'name': 'blue',
        'wins': 0
    }]

    for i in range(epochs):
        log('-' * 100)
        log('epoch: {}'.format(i + 1))
        game = Board()
        while not game.stale and not game.winner:
            # Exit if the board is full
            for bot in bots:
                winner = game.player_move(bot['mdl'].sym, *bot['mdl'].select_move(game.board))
                log('winner found:', winner)
                if winner:
                    optimize_bot(game, bot1, bot2)
                    break
                elif winner == 'draw':
                    optimize_bot(game, bot1, bot2)
                    break
    return bots[0]['wins'], bots[1]['wins']


def main():
    bot1 = Agent(sym='X')
    bot2 = Agent(sym='O')
    epochs = int(input('Enter the number of epochs for training: '))
    train(epochs, bot1, bot2)
    game = Board(player_sym='O')
    bot1.get_serious()
    while not game.stale:
        game.bot_play(*bot1.select_move(game.board))
        if game.winner:
            break
        raw_coords = input('Enter your coordinates (comma-separated):')
        coords = (int(value) for value in raw_coords.split(','))
        game.play(*coords)
        if game.winner:
            break


if __name__ == '__main__':
    main()
