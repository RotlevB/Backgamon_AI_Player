import sys
from src.game import Game
from src.colour import Colour
from src.strategies import MoveRandomPiece, MoveFurthestBackStrategy

def main():
    white_strategy = MoveRandomPiece()
    black_strategy = MoveFurthestBackStrategy()
    first_player = Colour.WHITE

    time_limit = input("Enter time limit in seconds (or 'inf' for no limit): ")
    if time_limit.lower() == 'inf':
        time_limit = -1
    else:
        time_limit = int(time_limit)

    game = Game(white_strategy, black_strategy, first_player, time_limit)
    game.run_game()

if __name__ == "__main__":
    main()
