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
g = [None]*10
g[0] = GameState(chess.Board('r1b3k1/6p1/P1n1pr1p/q1p5/1b1P4/2N2N2/PP1QBPPP/R3K2R'),0) # done
g[1] = GameState(chess.Board('2nq1nk1/5p1p/4p1pQ/pb1pP1NP/1p1P2P1/1P4N1/P4PB1/6K1'),1)
g[2] = GameState(chess.Board('8/3r2p1/pp1Bp1p1/1kP5/1n2K3/6R1/1P3P2/8'),1)
g[3] = GameState(chess.Board('8/4kb1p/2p3pP/1pP1P1P1/1P3K2/1B6/8/8'),1)
g[4] = GameState(chess.Board('b1R2nk1/5ppp/1p3n2/5N2/1b2p3/1P2BP2/q3BQPP/6K1'),1)
g[5] = GameState(chess.Board('3rr1k1/pp3pbp/2bp1np1/q3p1B1/2B1P3/2N4P/PPPQ1PP1/3RR1K1'),1)
g[6] = GameState(chess.Board('r1b1qrk1/1ppn1pb1/p2p1npp/3Pp3/2P1P2B/2N5/PP1NBPPP/R2Q1RK1'),0)
g[7] = GameState(chess.Board('2R1r3/5k2/pBP1n2p/6p1/8/5P1P/2P3P1/7K'),1)
g[8] = GameState(chess.Board('2r2rk1/1p1R1pp1/p3p2p/8/4B3/3QB1P1/q1P3KP/8'),1)
g[9] = GameState(chess.Board('r1bq1rk1/p4ppp/1pnp1n2/2p5/2PPpP2/1NP1P3/P3B1PP/R1BQ1RK1'),0)

from stockfish import Stockfish
stockfish = Stockfish("C://Users//jasak//MGR//Stockfish//stockfish_14.1_win_x64_popcnt", parameters={"Threads": 12, "Minimum Thinking Time": 60})
stockfish.set_depth(12)

###

a = []
fen = 'r3k2r/5ppp/p3p3/1p1p4/1PpP4/2P1P3/P3KPPP/RR6'
a.append(GameState(chess.Board(fen),1))
print(a[0].board)
print(a[0].board.fen())
stockfish.set_fen_position(a[0].board.fen())

print(stockfish.get_best_move())
## minimax eval

# for i, state in enumerate(g):
#     print(i+1)
#     bm = AI.minimax(state, 5, -inf, inf, True, 1)
#     print(bm)


## stockfish eval
#
# from stockfish import Stockfish
# stockfish = Stockfish("C://Users//jasak//MGR//Stockfish//stockfish_14.1_win_x64_popcnt", parameters={"Threads": 12, "Minimum Thinking Time": 60})
# # stockfish.set_elo_rating(2500)
# stockfish.set_depth(24)
# for i, state in enumerate(g):
#     print(i+1, end='\t')
#     fen = state.board.fen()
#     stockfish.set_fen_position(fen)
#     bm = stockfish.get_best_move()
#     print(bm)
#

# stockfish.set_fen_position("r1b3k1/6p1/P1n1pr1p/q1p5/1b1P4/2N2N2/PP1QBPPP/R3K2R b - - 0 1")

# print(stockfish.get_best_move())
# print(stockfish.get_evaluation())


# print(g[0].board.fen())
