import chess
# from chess import Board
import numpy as np
# import pandas as pd
import itertools
# import loggers as lg
import logging
order = ['K','Q','R','B','N','P','k','q','r','b','n','p']
import moves

# path_to_movelist = r"C:\Users\jasak\PycharmProjects\pythonProject\movelist.txt"

def array_to_board(helper):
    rows = []
    for s in helper:
        strng = []
        temp = 0
        for ind, char in enumerate(s):
            if char.isdigit():
                temp += 1
                if ind == len(s) - 1:
                    strng.append(str(temp))
            else:
                if temp:
                    strng.append(str(temp))
                    strng.append(char)
                    temp = 0
                else:
                    strng.append(char)

        rows.append(''.join(strng))

    state_in_string = '/'.join(rows)
    return chess.Board(state_in_string)


def tensor_to_board(tensor: np.ndarray):
    helper = np.ones((8, 8), dtype=np.str)
    tensor = tensor.reshape(12,8,8)
    for i in range(12):
        for x in range(8):
            for y in range(8):
                if tensor[i, x, y]:
                    helper[x, y] = order[i]

    helper = helper[::-1]
    print(helper)
    rows = []
    for s in helper:
        strng = []
        temp = 0
        for ind, char in enumerate(s):
            if char.isdigit():
                temp += 1
                if ind == len(s) - 1:
                    strng.append(str(temp))
            else:
                if temp:
                    strng.append(str(temp))
                    strng.append(char)
                    temp = 0
                else:
                    strng.append(char)

        rows.append(''.join(strng))

    state_in_string = '/'.join(rows)
    return chess.Board(state_in_string)


class Game:
    def __init__(self):
        self.currentPlayer = 1
        self.gameState = GameState(chess.Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'), 1)
        self.actionSpace = self.gameState.movelist
        self.pieceOrder = ['K','Q','R','B','N','P','k','q','r','b','n','p']
        self.pieces = {'1': "WHITE", '0': "BLACK"}
        self.input_shape = (13,8,8)
        self.grid_shape = (8, 8)
        self.name = 'chess'
        self.state_size = self.gameState.binary.shape
        self.action_size = len(self.actionSpace)

    def reset(self):
        self.gameState = GameState(chess.Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'), 1)
        self.currentPlayer = 1
        return self.gameState

    def step(self, action):

        next_state, value, done = self.gameState.takeAction(action)
        self.gameState = next_state
        # self.currentPlayer = -self.currentPlayer
        self.currentPlayer = 1 if self.currentPlayer == 0 else 0
        info = None
        return ((next_state, value, done, info))

    def step_s(self, action):

        next_state, value, done = self.gameState.takeAction_s(action)
        self.gameState = next_state
        # self.currentPlayer = -self.currentPlayer
        self.currentPlayer = 1 if self.currentPlayer == 0 else 0
        info = None
        return ((next_state, value, done, info))

    def identities(self, state, actionValues):

        identities = [(state, actionValues), (state, actionValues)]
        return identities


class GameState:

    def __init__(self, board: chess.Board, playerTurn):
        self.board = board
        self.pieceOrder = ['K','Q','R','B','N','P','k','q','r','b','n','p']
        self.pieces = {'1':"WHITE", '0': "BLACK"}
        self.playerTurn = playerTurn
        self.board.turn = playerTurn
        self.allowedActions = self._allowedActions()
        self.movelist = moves.movelist
        self.binary = self._binary()
        self.value = self._getValue()
        self.score = self._getScore()
        self.score_alpha = self._getScore_alpha()
        self.isEndGame = self._checkForEndGame()
        self.id = self._convertStateToId()

    # def copy(self):
    #     return GameState(self.board.copy(), self.playerTurn.copy())

    def board_to_tensor(self):
        order_piecenames = [chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
        tensor = np.zeros((12, 8, 8))
        for ind, piece in enumerate(order_piecenames):
            tensor[ind, :, :] = np.array(self.board.pieces(piece, chess.WHITE).tolist(), dtype=np.int).reshape(8, 8)
            tensor[ind + 6, :, :] = np.array(self.board.pieces(piece, chess.BLACK).tolist(), dtype=np.int).reshape(8, 8)
        return tensor

    def _allowedActions(self) -> list:
        actions = list(self.board.legal_moves)
        # ml = moves.movelist
        indexes = [(np.where(moves.movelist == str(action))[0]) for action in actions]
        chain = list(itertools.chain.from_iterable(indexes))
        return chain

    def actions(self):
        # return list(self.board.legal_moves)
        return [str(move) for move in list(self.board.legal_moves)]

    # def _read_movelist(self) -> np.ndarray:
    #     return np.squeeze(pd.read_csv(path_to_movelist, delimiter='\n', header=None).values)

    def _binary(self) -> np.ndarray:
        order_piecenames = [chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
        # tensor = np.zeros((2, 6, 8, 8))
        tensor = np.zeros((13, 8, 8))
        white = np.zeros((6, 8, 8))
        black = np.zeros((6, 8, 8))
        for ind, piece in enumerate(order_piecenames):
            white[ind] = np.array(self.board.pieces(piece, chess.WHITE).tolist(), dtype=np.int).reshape(8, 8)
            black[ind] = np.array(self.board.pieces(piece, chess.BLACK).tolist(), dtype=np.int).reshape(8, 8)

        tensor[0, :, :].fill(self.playerTurn)
        tensor[1:7, :, :] = white if self.playerTurn else black
        tensor[7:13, :, :] = black if self.playerTurn else white

        if self.playerTurn == 0:
            tensor = np.flip(tensor, axis=1)
            tensor = np.flip(tensor, axis=2)

        return tensor

    def _convertStateToId(self):
        # return self.board.epd()[:-6]
        return self.board.fen()

    def _checkForEndGame(self) -> int:

        conditions = np.array([self.board.is_checkmate(), self.board.is_stalemate(), self.board.is_insufficient_material()])
        if np.any(conditions):
            return 1
        else:
            return 0

    # def _getValue(self) -> tuple:
    #     conditions = np.array([self.board.is_checkmate(), self.board.is_stalemate(), self.board.is_insufficient_material()])
    #     if np.any(conditions):
    #         return (-1, -1, -1)
    #     else:
    #         return (0, 0, 0)

    def get_leader(self):
        scores = np.array([4,9,5,3,3,1])
        white = np.sum(np.array([np.sum(piece) for piece in self.board_to_tensor()[:6]]) * scores)
        black = np.sum(np.array([np.sum(piece) for piece in self.board_to_tensor()[6:12]]) * scores)
        if white > black:
            return 1
        elif black > white:
            return 0
        else:
            return -1
        
    def get_value_alpha(self):
        leader = self.get_leader()
        if self.playerTurn == 0 and leader == 1:
            return (-1,-1,-1)
        elif self.playerTurn == 1 and leader == 0:
            return (-1,-1,-1)
        elif self.playerTurn == 1 and leader == 1:
            return (1,1,1)
        elif self.playerTurn == 0 and leader == 0:
            return (1,1,1)
        else:
            return (0,0,0)
        
    def _getValue(self) -> tuple:
        res = self.board.result()
        if self.playerTurn == 1 and res == "0-1":
            return (-1,-1,-1)
        elif self.playerTurn == 0 and res == "1-0":
            return (-1,-1,-1)
        else:
            return (0,0,0)

    def _getScore_alpha(self) -> tuple:
        tmp = self.get_value_alpha()
        #zmieniono z tmp = self.value
        return(tmp[1],tmp[2])

    def _getScore(self) -> tuple:
        # tmp = self.get_value_alpha()
        tmp = self.value
        return(tmp[1],tmp[2])

    def takeAction(self,action):
        newBoard = self.board.copy()
        newBoard.push(chess.Move.from_uci(self.movelist[action]))
        turn = 1 if self.playerTurn == 0 else 0
        newState = GameState(newBoard,turn)
        del newBoard
        value=0
        done=0
        if not newState.board.is_valid():
            done = -1
            return(newState,value,done)

        if newState.isEndGame:
            value = newState.value[0]
            done=1
        return (newState,value,done)

    def takeAction_s(self,action):
        newBoard = self.board.copy()
        newBoard.push(chess.Move.from_uci(action))
        turn = 1 if self.playerTurn == 0 else 0
        newState = GameState(newBoard,turn)
        del newBoard
        value=0
        done=0
        if not newState.board.is_valid():
            done = -1
            return(newState,value,done)

        if newState.isEndGame:
            value = newState.value[0]
            done=1
        return (newState,value,done)

    def render(self,logger):
        # logger.info("\n" + str(self.board))
        # logger.info('-----------------')
        logger.info(str(self.board.fen()))

