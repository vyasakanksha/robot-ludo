"""
Created on Fri Oct 23 14:00:00 2020

@author: abhishekroy
"""

import random
from state import Board, getmoves, makemove

def make_start_board(counters,length,safesquares=None):
    safe = [0]*length
    safe[0],safe[length//2] = 1,1
    if safesquares:
        for s in safesquares:
            safe[s] = 1
            
    return Board(counters=counters,length=length,safe=safe,red=[0]*length,blue=[0]*length,
                redpen=counters,bluepen=counters)


def match(counters,length,safesquares,player1,player2,maxiters=1000,starting=0,
          seed=None,verbose=False):
    '''
    Returns winner of a match between player1 (0) and player2 (1).
    The colours are RED and BLUE respectively.
    '''
    if seed is not None:
        random.seed(seed)
    b = make_start_board(counters,length,safesquares)
    result = None
    cuts = 0
#     movelist = []
    for i in range(maxiters):
        turn = (i + starting) % 2
        roll = random.randint(1,6)
        player = (player1,player2)[turn]
        move = player(b,turn,roll)
        
        rp,bp = b.redpen,b.bluepen
        b = makemove(b,move,copyboard=False)
        
        # if verbose:
        #     print(turn,roll,move)
        if b.redpen>rp or b.bluepen>bp:
            cuts+=1

        if b.redpen == 0 and sum(b.red)==0:
            result = 0
            break
        if b.bluepen == 0 and sum(b.blue)==0:
            result = 1
            break
        
#         movelist += [move]
    if verbose:
        print("Cuts",cuts,"Moves",i+1)
    return result

def judge(counters,length,safesquares,player1,player2,games=100,starting=0):
    '''
    Returns winning fractions (frac1,frac2) 
    '''
    results = [match(counters,length,safesquares,player1,player2,
                     maxiters=1000,starting=starting)
               for _ in range(games)]
    results = [r for r in results if r is not None]
    avg2 = sum(results)/len(results)

    return 1-avg2, avg2


def simple_player(board,turn,roll):
    moves = getmoves(board,turn,roll)
    mymove = moves[0]
    return mymove


def halfshift(x, L):
    if x >= L//2:
        return x - L//2
    else:
        return x + L//2


def simple_player2(board,turn,roll):
    '''Symmetric'''
    moves = getmoves(board,turn,roll)
    if (turn,0,0) in moves:
        mymove = (turn,0,0) 
    elif turn == 0:
        mymove = moves[0]
    elif turn == 1:
        mymove = min(moves, key=lambda triple: halfshift(triple[1],board.length))
    return mymove

def simple_player3(board,turn,roll):
    moves = getmoves(board,turn,roll)
    mymove = random.choice(moves)
    return mymove

def simple_player4(board,turn,roll):

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

def simple_player5(board,turn,roll):
    rlist = []
    moves = getmoves(board,turn,roll)
    for m in moves:
        b = makemove(board,m,copyboard=True)
        rlist.append(reward1(board, b, turn))
    m = max(rlist)
    optim_moves = [moves[i] for i, j in enumerate(rlist) if j == m]
    mymove = random.choice(optim_moves)
    bb = makemove(board,mymove,copyboard=True)
    return mymove

def reward_temp(counters, pen, inplay, safe):
    r = 0
    r += sum(inplay)*5
    r += (counters - (pen + sum(inplay))) * 20

    r += sum([x for x,y in zip(inplay,safe) if y]) * 5
    # r += (counters - (pen + inplay)) * 20
    return r

def reward_temp1(counters, pen, inplay, safe):
    r = 0
    r += sum(inplay) * 2
    r += (counters - (pen + sum(inplay))) * 10
    
    # r += (counters - (pen + inplay)) * 20
    r += sum([x for x,y in zip(inplay,safe) if y]) * 0
    return r

def reward(oldBoard, board, turn):
    reward = 0
    if turn == 0:
        reward = (reward_temp(board.counters, board.redpen, board.red, board.safe)) - (reward_temp(board.counters, board.bluepen, board.blue, board.safe))
    if turn == 1:
        reward = (reward_temp(board.counters, board.bluepen, board.blue, board.safe)) - (reward_temp(board.counters, board.redpen, board.red, board.safe))
    return reward

def reward1(oldBoard, board, turn):
    reward1 = 0
    if turn == 0:
        reward1 = (reward_temp1(board.counters, board.redpen, board.red, board.safe)) - (reward_temp1(board.counters, board.bluepen, board.blue, board.safe))
    if turn == 1:
        reward1 = (reward_temp1(board.counters, board.bluepen, board.blue, board.safe)) - (reward_temp1(board.counters, board.redpen, board.red, board.safe))
    return reward1

def extract_features(self, player):
    features = []
    for p in self.players:
        for col in self.grid:
            feats = [0.] * 6
            if len(col) > 0 and col[0] == p:
                for i in range(len(col)):
                    feats[min(i, 5)] += 1
            features += feats
        print(float(len(self.bar_pieces[p])) / 2.)
        features.append(float(len(self.bar_pieces[p])) / 2.)
        features.append(float(len(self.off_pieces[p])) / self.num_pieces[p])
    if player == self.players[0]:
        features += [1., 0.]
    else:
        features += [0., 1.]
    return np.array(features).reshape(1, -1)

