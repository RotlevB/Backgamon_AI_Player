'''
created by: Benjamin Rotlevy
last update: 16.12.2024
'''
from time import sleep
from tkinter import *
from AI_Player import *
from Huristic_AI_Player import *
import random
from Backgammon_Game import roll

TRI_HEIGHT = 200
TRI_WIDTH = 50

class BackgammonAIvsAI:

    def __init__(self, window):
        self.black_player = Huristic_AI_Player()
        self.white_player = AI_Player("white")
        self.window = window

        self.title = StringVar()
        self.title.set("AI vs AI: Let the game begin!")

        self._canvas = Canvas(self.window, width=13 * TRI_WIDTH, height=3 * TRI_HEIGHT)
        self._canvas.pack()
        self.turnLabel = Label(self.window, textvariable=self.title)
        self.turnLabel.pack()

        self.render()
        self.play_game()

    def render(self):
        '''Renders the game board every 50 milliseconds'''
        self._canvas.delete('piece')
        for player, color in [(self.black_player, 'black'), (self.white_player, 'white')]:
            pieces = player.get_pieces()
            unique_pieces = set(pieces)
            for piece in unique_pieces:
                count = pieces.count(piece)
                for i in range(count):
                    x = 12 - piece if piece <= 12 else piece - 13
                    y = i if piece <= 12 else int(self._canvas.cget('height')) - (i + 1) * TRI_WIDTH
                    self._canvas.create_oval(x * TRI_WIDTH, y, (x + 1) * TRI_WIDTH, y + TRI_WIDTH, fill=color, tags='piece')
        self._canvas.after(50, self.render)

    def play_game(self):
        while not (self.black_player.win() or self.white_player.win()):
            self.title.set("Black's turn")
            self.window.update()
            black_roll = roll()
            self.perform_turn(self.black_player, self.white_player, black_roll, "black")

            if self.black_player.win():
                self.title.set("Black wins!")
                break
            sleep(0.3)
            self.title.set("White's turn")
            self.window.update()
            white_roll = roll()
            self.perform_turn(self.white_player, self.black_player, white_roll, "white")

            if self.white_player.win():
                self.title.set("White wins!")
                break
            sleep(0.3)

    def perform_turn(self, current_player, opponent, dice_roll, color):
        board = self.status_format()
        print("board: ", board)
        try:
            current_player.play(board, dice_roll, color)
            opponent.set_pieces(current_player.get_other_pieces())
        except ValueError:
            print("No valid moves for {} player".format(color))
            pass  # Skip turn if no valid moves

    def status_format(self):
        board = [0] * 28
        print("white pieces: ", self.white_player.get_pieces())
        for point in self.white_player.get_pieces():
            if point == 0:  # white captured
                board[26] += 1
            elif point == 25:
                board[24] += 1  # white out
            else:
                board[point - 1] += 1
        print("black pieces: ", self.black_player.get_pieces())
        for point in self.black_player.get_pieces():
            if point == 25:  # black captured
                board[27] += 1
            elif point == 0:
                board[25] += 1  # black out
            else:
                board[point - 1] -= 1

        return board
if __name__ == '__main__':
    root = Tk()
    root.title('Backgammon AI vs AI')
    app = BackgammonAIvsAI(root)
    root.mainloop()
