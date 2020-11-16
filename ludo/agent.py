from state import Board, makemove, getmoves, isvalidboard, isvalidmove
import random
import numpy as np


def random_agent(board,turn,roll):
    moves = getmoves(board,turn,roll)
    mymove = random.choice(moves)
    return mymove

def akanksha(board,turn,roll):
    rlist = []
    moves = getmoves(board,turn,roll)
    for m in moves:
        b = makemove(board,m,copyboard=True)
        rlist.append(reward(board, b, turn))
    m = max(rlist)
    optim_moves = [moves[i] for i, j in enumerate(rlist) if j == m]
    mymove = random.choice(optim_moves)
    bb = makemove(board,mymove,copyboard=True)
    
    return mymove

def reward_temp(counters, pen, inplay, safe):
    r = 0
    r += (counters - pen)*5 # 5 points for counters out of the pen
    r += (counters - (pen + sum(inplay))) * 20 # 20 points for counters which have made it home

    safe_list = list(filter(lambda x: x > 1, [a + b - 1 for a, b in zip(safe, inplay)]))
    r += sum(safe_list) * 10 # 10 points for counters which have made it home
    return r

def reward(oldBoard, board, turn):
    reward = 0
    if turn == 0:
        reward = (reward_temp(board.counters, board.redpen, board.red, board.safe)) - (reward_temp(board.counters, board.bluepen, board.blue, board.safe))
    if turn == 1:
        reward = (reward_temp(board.counters, board.bluepen, board.blue, board.safe)) - (reward_temp(board.counters, board.redpen, board.red, board.safe))
    return reward