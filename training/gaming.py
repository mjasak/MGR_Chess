import numpy as np

import chess
import AI
import sys
import os
from math import inf
import time
from game import Game, GameState

env = Game()
done = 0
state = env.reset()
turn = 0
# while done == 0:
for i in range(1800):
    turn = turn + 1
    # print("turn: ", turn)

    # player[state.playerTurn].act(state,..)
    # 1. choose action
    # action, value = AI.minimax(state,3,-inf,inf,True, state.playerTurn)
    action = AI.random_move(state)
    print(f"Turn: {turn} \t Action: {action}")
    # print(f"playerturn: {int(state.playerTurn)}")
    # 2. commit the action to game
    state.board.push_uci(action)
    state.playerTurn = state.board.turn
    # print(state.board, flush=False)
    # print(f"action pushed")
    # print(f"playerturn: {int(state.playerTurn)}")
    if state.isEndGame or len(state.actions()) == 0:
        print(f"winner: {state.playerTurn}")
        break
    # 3. check for endgame
    # 4. print the winner
if state.get_leader() == 1:
    winner_print = "white"
elif state.get_leader() == 0:
    winner_print = "black"
else:
    winner_print = "draw"
print("winner: ", winner_print)
print("Board: ", state.board.fen())
