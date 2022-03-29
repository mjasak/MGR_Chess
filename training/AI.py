import random
from math import inf

from game import Game, GameState
import chess
import moves as mv
from copy import deepcopy
import numpy as np
import mcts_stat as mc

def get_scores(state: GameState) -> tuple:
    scores = np.array([200,9,5,3,3,1])
    currentMobility = len(state.allowedActions)
    state2 = deepcopy(state)
    state2.playerTurn = 0 if state.playerTurn == 1 else 1
    opponentMobility = len(state2.allowedActions)

    if state.playerTurn:
        white = np.sum(np.array([np.sum(piece) for piece in state.board_to_tensor()[:6]]) * scores) + 0.5*currentMobility
        black = np.sum(np.array([np.sum(piece) for piece in state.board_to_tensor()[6:12]]) * scores) + 0.5*opponentMobility
        if state.board.is_check():
            white = white + 10
        if state.board.is_checkmate():
            white = white + 1000
        # ratio = white/black
    else:
        white = np.sum(
            np.array([np.sum(piece) for piece in state.board_to_tensor()[:6]]) * scores) + 0.1 * opponentMobility
        black = np.sum(
            np.array([np.sum(piece) for piece in state.board_to_tensor()[6:12]]) * scores) + 0.1 * currentMobility
        if state.board.is_check():
            black = black + 10
        if state.board.is_checkmate():
            black = black + 1000
        # ratio = black/white
    return white,black


def random_move(state: GameState) -> str:
        # moves = state.allowedActions
        moves = state.actions()
        if moves:
            return random.choice(moves)


def evaluate(state: GameState, playerTurn: bool) -> float:
    w,b = get_scores(state)
    if playerTurn == 1:
        return w-b
    else:
        return b-w


def minimax(state,depth,alpha,beta,maximizing_player, maximizing_color):

    if depth == 0 or state.isEndGame or len(state.actions()) == 0:
        return None, evaluate(state,maximizing_color)

    moves = state.actions()
    # print(len(moves))
    best_move = random.choice(moves)
    # print(best_move)

    if maximizing_player:
        max_eval = -inf
        max_eval_list = []
        # state_copy = state.deepcopy()
        for move in moves:
            # print(move)
            # print(state)
            state.board.push(chess.Move.from_uci(move))
            current_eval = minimax(state, depth-1, alpha, beta, False, maximizing_color)[1]
            state.board.pop()
            if current_eval > max_eval:
                max_eval = current_eval
                best_move = move
            #     max_eval_list = []
            # elif current_eval == max_eval:
            #     max_eval_list.append(best_move)
            # print(f"alpha:{alpha}")
            alpha = max(alpha,current_eval)
            if beta <= alpha:
                # print(f"beta:{beta}\t alpha:{alpha}")
                break
        # if len(max_eval_list) > 1:
        #     # print(len(max_eval_list))
        #     # print(max_eval)
        #     best_move = random.choice(max_eval_list)
        return best_move, max_eval
    else:
        min_eval = inf
        # state_copy = state.deepcopy()
        min_eval_list = []
        for move in moves:
            # print(move)
            # print(state.board.fen())
            # print(state)
            state.board.push(chess.Move.from_uci(move))
            current_eval = minimax(state, depth - 1, alpha, beta, True, maximizing_color)[1]
            state.board.pop()
            if current_eval < min_eval:
                min_eval = current_eval
                best_move = move
            #     min_eval_list = []
            # elif current_eval == min_eval:
            #     min_eval_list.append(best_move)
            beta = min(beta, current_eval)
            if beta <= alpha:
                # print(f"beta:{beta}\t alpha:{alpha}")
                break
        # if len(min_eval_list) > 1:
        #     best_move = random.choice(min_eval_list)
        return best_move, min_eval


def alphamax(state, mcts_sims, mcts=None, cpuct=1) -> tuple:
    def build_mcts():
        root = mc.Node(state)
        mctree = mc.MCTS(root, cpuct)
        return mctree

    def changeRootMCTS():
        mcts.root = mcts.tree[state.id]

    def simulate():
        ##### MOVE THE LEAF NODE
        leaf, value, done, breadcrumbs = mcts.moveToLeaf()
        ##### EVALUATE THE LEAF NODE
        value, breadcrumbs = evaluateLeaf(leaf, value, done, breadcrumbs)

        ##### BACKFILL THE VALUE THROUGH THE TREE
        mcts.backFill(leaf, value, breadcrumbs)

    def evaluateLeaf(leaf, value, done, breadcrumbs):
        if done == 0:
            value = np.tanh(evaluate(leaf.state, leaf.state.playerTurn))
            allowedActions = leaf.state.actions()

            for idx, action in enumerate(allowedActions):
                newState, _, _ = leaf.state.takeAction_s(action)
                if newState.id not in mcts.tree:
                    node = mc.Node(newState)
                    mcts.addNode(node)
                else:
                    node = mcts.tree[newState.id]
                # newEdge = mc.Edge(leaf, node, probs[idx], action)
                newEdge = mc.Edge(leaf, node, 0, action)
                leaf.edges.append((action, newEdge))

        else:
            print("done")

        return ((value, breadcrumbs))

    def chooseAction(pi, values):
        actions = np.argwhere(pi == max(pi))
        action = random.choice(actions)[0]

        value = values[action]

        return mv.movelist[action], value

    def getAV():
        edges = mcts.root.edges
        pi = np.zeros(4096, dtype=np.integer)
        values = np.zeros(4096, dtype=np.float32)

        for action, edge in edges:
            ind = np.where(mv.movelist == action)[0]
            pi[ind] = edge.stats['N']
            values[ind] = edge.stats['Q']

        return pi, values

    if mcts is None or state.id not in mcts.tree:
        mcts = build_mcts()
    else:
        changeRootMCTS()
    print("here we have mcts tree")
    print("starting simulations")
    for sim in range(mcts_sims):
        simulate()
    print("ended simulations")
    print("getting action values")
    #### get action values
    pi, values = getAV()
    print("got action values")
    print("choosing action")
    ####pick the action
    action, value = chooseAction(pi, values)
    print("the wise had spoken, action has been chosen")
    # print(moves.movelist[action])
    return action, value



