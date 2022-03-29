from gen_random_fen import generate
import csv
import scipy as sc
import numpy as np
from math import exp, factorial
import chess
import random
from stockfish import Stockfish
from moves import movelist
import pandas as pd
import csv


def generate_board():
    count = 1
    brd = chess.Board(generate())
    valid = brd.is_valid()
    while not valid:
        count += 1
        brd = chess.Board(generate())
        valid = brd.is_valid()
    return brd


def pois(n, lam):
    return [lam ** k * exp(-lam) / factorial(k) for k in range(n)]


def eval_stock(board, eng):
    eng.set_fen_position(board.fen())
    n = len(list(board.legal_moves))
    if n:
        return eng.get_top_moves(n)
    else:
        return None


def sample(board, debug=False):
    ev = eval_stock(board, stockfish)
    if ev:
        pdist = pois(len(ev), 0.7)
        if debug:
            print(len(ev))
            print(board.fen())
            print(list(board.legal_moves))
        it = 0
        turn = 1 if board.fen().split(" ")[-5] == 'w' else 0
        avs = [0]*4096
        for item in ev:
            if item['Mate'] is not None:
                value = 1 if item['Mate'] > 0 else -1  # negative values favour black
                av = pdist[it]
                it += 1
            elif item['Centipawn'] is not None:
                value = 1 if item['Centipawn'] >= 0 else -1  # negative values favour black
                av = pdist[it]
                it += 1

            if turn == 0:
                try:
                    value = -value
                except UnboundLocalError:
                    print("UnboundLocalError")
                    print(board.fen())
                    print(item)
                    break
            try:
                item['Move'] = item['Move'][:4]
                index = np.where(movelist == item['Move'])[0][0]
            except IndexError:
                print("IndexError")
                print("np.where(...)[0] = ", np.where(movelist == item['Move'][0]))
                print(item)
            try:
                avs[index] = av
            except UnboundLocalError:
                print("UnboundLocalError")
                print(board.fen())
                print(item)
                break

        row = {'fen': board.fen(), 'turn': turn, 'value': value, "AV": avs}
        return row
    else:
        pass


stockfish = Stockfish("C://Users//jasak//MGR//Stockfish//stockfish_14.1_win_x64_popcnt", parameters={"Threads": 12, "Minimum Thinking Time": 60})
# board = generate_board()
# print(board.fen())
# print(board.status())
# print(list(board.legal_moves))
# d = eval_stock(board,stockfish)
# print(d)
# row = []
# pdist = pois(len(d), 1)

a1 = chess.Board("3q4/3b1Kb1/1qrkp3/6p1/5b2/2n3r1/2b1N2r/1r4b1 b - - 0 1")

# print(board.fen().split(" ")[-5])
# print(ev)

# with open('fens.txt','w') as f:
#     for _ in range(10):
#         f.write(sample(generate_board())['fen'])
#         f.write("\n")

f = open('fens.txt','w')
g = open('turns.txt','w')
h = open('values.txt','w')
k = open('avs.txt','w')

# with open('fens.csv', 'w', newline='') as fens:
#     writer_fens = csv.writer(fens, delimiter=' ')
#     with open('fens.csv', 'w', newline='') as turns:
#         writer_turns = csv.writer(turns, delimiter=' ')
#         with open('values.csv', 'w', newline='') as values:
#             writer_values = csv.writer(values, delimiter=' ')
#             with open('avs.csv', 'w', newline='') as avs:
#                 writer_avs = csv.writer(avs, delimiter=' ')
for i in range(100_000):
    a = generate_board()
    s = sample(a)

    if s:
        f.write(a.fen())
        f.write("\n")
        g.write(str(s['turn']))
        g.write("\n")
        h.write(str(s['value']))
        h.write("\n")
        k.write(str(s['AV'])[1:-1])
        k.write("\n")

                        # writer_fens.writerow(a.fen())
                        # writer_turns.writerow(str(s['turn']))
                        # writer_values.writerow(str(s['value']))
                        # writer_avs.writerow(s['AV'])


f.close()
g.close()
h.close()
k.close()

