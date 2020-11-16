from state import Board, makemove, getmoves, isvalidboard, isvalidmove
from draw import drawboard
from match import make_start_board, match, judge, simple_player3, simple_player4, simple_player5
import random
from agent import akanksha
from model import tdl_agent


# print(match(10,12,[3,9],simple_player4,simple_player5,verbose=True))
print(judge(10, 20, None, tdl_agent,  akanksha, games=1000, starting=0))
