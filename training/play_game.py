import numpy as np

import chess
import AI
import os
from math import inf
import time

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# print(os.getcwd())
from game import Game, GameState
# g = GameState(chess.Board('r1bqkb1r/p2p2pp/n1p2p2/1p5n/P3NPQP/1P1PP3/2P3B1/R1B1K1NR b KQkq - 0 1'),1)
g = GameState(chess.Board('rn1r2k1/1pq2p1p/p2p1bpB/3P4/P3Q3/2PB4/5PPP/2R1R1K1'),1)
# print(g.board)
print("MINIMAX")
start_time = time.time()
bm = AI.minimax(g,3,-inf,inf,True,1)
print(bm)
end_time = time.time()
print(f"Execution: {end_time-start_time}")
print("\n")
# print("ALPHAMAX")
# start_time = time.time()
# bm = AI.alphamax(g,100)
# print(bm)
# end_time = time.time()
# print(f"Execution: {end_time-start_time}")
# sss = GameState(chess.Board('r1bqkb1r/p2p2pp/n1p2p2/1p5n/P3NPQP/1P1PP3/2P3B1/R1B1K1NR b KQkq - 0 1'),1)
# sss.board.push(chess.Move.from_uci("g4g7"))
