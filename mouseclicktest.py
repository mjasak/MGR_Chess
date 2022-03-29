from stockfish import Stockfish
import chess


stockfish = Stockfish(r'C://Users//jasak//MGR//Stockfish//stockfish_14.1_win_x64_popcnt',parameters={"Threads": 2, "Minimum Thinking Time": 60} )

# fen = '5r1k/1R4p1/1R5p/p2p4/4n3/4Q1P1/2q2P1P/6K1 b - - 0 1'
# fen = 'k7/pp1K4/N7/8/8/8/8/7B w - - 0 1'
m3_1 = '1Q6/8/7k/3p4/2p3bp/2P1K1n1/1P4P1/5r2 b - - 0 0'
g3f5
# a = chess.Board(fen)
# print(a)
stockfish.set_depth(24)
stockfish.set_fen_position(m3_1)
print(stockfish.get_best_move())
