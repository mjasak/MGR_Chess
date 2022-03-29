import chess
from stockfish import Stockfish
from chess import Board
import chess.pgn


def move(fen,sf):
    sf.set_fen_position(fen)
    return sf.get_best_move()


def main():
    board = chess.Board()
    sf1 = Stockfish(r'C://Users//jasak//MGR//Stockfish//stockfish_14.1_win_x64_popcnt')
    sf2 = Stockfish(r'C://Users//jasak//MGR//Stockfish//stockfish_14.1_win_x64_popcnt')
    sf1.set_depth(18)
    sf1.set_elo_rating(3000)
    sf2.set_depth(12)
    sf2.set_elo_rating(1000)
    game = chess.pgn.Game()
    game.headers["Event"] = "Example"
    f = open("fens_match.txt", 'w')
    g = open("actions_match.txt", 'w')
    # p = open("pgn_match.txt", 'w')

    while True:
        if board.turn == 1:
            action = move(board.fen(), sf1)
        else:
            action = move(board.fen(), sf2)

        if board.is_game_over():
            break

        f.write(board.fen())
        f.write("\n")
        g.write(action)
        g.write("\n")
        print(action)
        if board.fen() == 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1':
            node = game.add_variation(chess.Move.from_uci(action))
        else:
            node = node.add_variation(chess.Move.from_uci(action))
        board.push_uci(action)

    print(board.fen())
    print(game)
    f.close()
    g.close()


if __name__ == '__main__':
    main()