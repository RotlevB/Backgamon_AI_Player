'''Creates a class for Human backgammon pieces
Created 2024
@author: Anni Ainesaz
'''
from tkinter import *
from AI_Player import *
from Human_Player import *
from Huristic_AI_Player import *
import random
from Backgammon_Game import roll

TRI_HEIGHT = 200
TRI_WIDTH = 50

class BackgammonGame:

    def __init__(self, window):
        self.turnir = []
        self.turnir.append(Huristic_AI_Player())
        self.turnir.append(Human_Player())
        self.scores = [0] * len(self.turnir)

        self.window = window

        for i in range(len(self.turnir)):
            for j in range(i + 1, len(self.turnir)):
                if self.start_game(self.turnir[i], self.turnir[j]) == "black":
                    self.scores[i] += 1
                else:
                    self.scores[j] += 1

        print("scores: ", self.scores)
        winner_idx = self.scores.index(max(self.scores))



    def start_game(self, black_player, white_player):
        self.black = black_player
        self.white = white_player

        self._rgrid = Grid()
        self._canvas = Canvas(self.window, width=13 * TRI_WIDTH, height=3 * TRI_HEIGHT)

        self.rolls = StringVar()
        self.roll_frame = Frame(self.window)

        self.title = StringVar()
        self.title.set("It's your turn! Roll the dice!")
        self.move_format_white = []
        self.move_format_black = []

        # timer
        self.time_remaining = StringVar()
        self.timer_label = Label(self.window, textvariable=self.time_remaining)
        self.time_left = 500  # Default turn time limit in seconds

        self.rollButton = Button(self.roll_frame, text='Roll', command=self.roll)
        self.dieLabel = Label(self.roll_frame, textvariable=self.rolls)
        self.turnLabel = Label(self.window, textvariable=self.title)
        self.endButton = Button(self.roll_frame, text='End Turn (if stuck)', command=self.end_turn)

        for space in range(12):
            self._canvas.create_polygon(space * TRI_WIDTH, 0, (space + 1) * TRI_WIDTH, 0, (space + .5) * TRI_WIDTH,
                                        TRI_HEIGHT, fill='#C19A6B')
        for s in range(12):
            # Found .cget() function online at bytes.com
            self._canvas.create_polygon(s * TRI_WIDTH, int(self._canvas.cget('height')), (s + 1) * TRI_WIDTH,
                                        int(self._canvas.cget('height')), (s + .5) * TRI_WIDTH,
                                        int(self._canvas.cget('height')) - TRI_HEIGHT, fill='#C19A6B')

        self.rollButton.pack(side=LEFT, anchor=W)
        self.dieLabel.pack(anchor=N)
        self.endButton.pack(anchor=E)
        self.roll_frame.pack()
        self._canvas.pack()
        self.turnLabel.pack()

        self.timer_label.pack()  # Add timer label to the UI

        self._canvas.bind('<Button-1>', self.whiteMove1)
        self._canvas.bind('<Button-3>', self.whiteMove2)

        self.render()

    def status_format(self):
        board = [0] * 28
        for point in self.white.get_pieces():
            if point == 0: # white captured
                board[26] += 1
            elif point == 25:
                board[24] += 1 # white out
            else:
                board[point - 1] += 1

        for point in self.black.get_pieces():
            if point == 25:  # black captured
                board[27] += 1
            elif point == 0:
                board[25] += 1  # black out
            else:
                board[point - 1] -= 1


        return board




    def roll(self, start_timer=True):
        self.r = roll()
        self.r.sort()
        rolled = self.r[:]
        rolled = str(rolled)
        rolled = rolled[1:-1]
        rolled = rolled.replace(',', '')
        self.rolls.set(rolled)
        self.rollButton.config(state = DISABLED)
        self.title.set('Choose a piece to move')
        if start_timer:
            self.start_timer()  # Start the timer for this turn

    def start_timer(self): #timer
        """Start or reset the turn timer."""
        self.time_left = 500  # Set the time limit (30 seconds per turn)
        self.update_timer()

    def update_timer(self): #timer
        """Update the timer and enforce turn end if time runs out."""
        if self.time_left > 0:
            self.time_remaining.set(f"Time left: {self.time_left} seconds")
            self.time_left -= 1
            self.window.after(1000, self.update_timer)
        else:
            self.title.set("Time's up! Ending your turn.")
            self.end_turn()  # Automatically end turn when time runs out

    def render(self):
        '''Renders the game board every 50 milliseconds'''
        self._canvas.delete('piece')
        bp = self.black.get_pieces()
        bp_nodups = bp[:]
        bp_nodups = list(set(bp_nodups))
        for piece in bp_nodups:
            idx = [i for i,x in enumerate(bp) if x==piece]
            if piece <= 12:
                for pos in range(len(idx)):
                    self._canvas.create_oval((12 - piece) * TRI_WIDTH, pos * TRI_WIDTH, (13 - piece) * TRI_WIDTH, (pos + 1) * TRI_WIDTH, fill = 'black', tags = 'piece')
            elif piece < 25:
                for pos in range(len(idx)):
                    self._canvas.create_oval((piece - 12) * TRI_WIDTH, int(self._canvas.cget('height')) - ((pos + 1) * TRI_WIDTH), (piece - 13) * TRI_WIDTH, int(self._canvas.cget('height')) - (pos * TRI_WIDTH), fill = 'black', tags = 'piece')
        wp = self.white.get_pieces()
        wp_nodups = wp[:]
        wp_nodups = list(set(wp_nodups))
        for piece in wp_nodups:
            #The following line was taken almost straight from stackoverflow (http://stackoverflow.com/questions/9542738/python-find-in-list)
            idx = [i for i,x in enumerate(wp) if x==piece]
            if piece <= 12:
                for pos in range(len(idx)):
                    self._canvas.create_oval((12 - piece) * TRI_WIDTH, pos * TRI_WIDTH, (13 - piece) * TRI_WIDTH, (pos + 1) * TRI_WIDTH, fill = 'white', tags = 'piece')
            elif piece < 25:
                for pos in range(len(idx)):
                    self._canvas.create_oval((piece - 12) * TRI_WIDTH, int(self._canvas.cget('height')) - ((pos + 1) * TRI_WIDTH), (piece - 13) * TRI_WIDTH, int(self._canvas.cget('height')) - (pos * TRI_WIDTH), fill = 'white', tags = 'piece')
        self._canvas.after(50, self.render)

    def whiteMove1(self, event):
        '''checks if the selected piece is valid to move'''
        die = self.rolls.get()
        self.select(event)
        piece = self.selected
        if piece not in self.white.get_pieces():
            self.title.set("That's an invalid piece to pick")
            self.rollButton.config(state=NORMAL)
            return
        self.title.set('Choose a position to move it to (right click)')

    def select(self, event):
        '''Selects a piece clicked on'''
        x = event.x
        y = event.y
        x = x // TRI_WIDTH
        y = y // TRI_HEIGHT
        if y == 0:
            self.selected = 12 - x
        elif y == 1:
            self.selected = 0
        else:
            self.selected = 13 + x

    def has_no_lower_points(self, pieces, selected_point, dice_roll):
        """
        Checks if there are no white pieces in points less than the selected point.
        """
        lower_points = [piece for piece in pieces if piece < selected_point]
        return all(piece > dice_roll for piece in lower_points)

    def whiteMove2(self, event):
        '''Takes the selected piece and moves it to the destination right-clicked by user'''
        self.goto(event)

        # if self.selected == 0:
        #     self.move_format_white.append([27, self.destination - 1])
        # elif self.destination == 25:
        #     self.move_format_white.append([self.selected - 1, 25])
        # else:
        #     self.move_format_white.append([self.selected - 1, self.destination - 1])


        distance = self.destination - self.selected
        r = self.rolls.get()
        r = r.split()

        is_less = False
        to_remove = max(r)
        for num in r:
            if int(num) >= distance:
                is_less = True
                to_remove = min(int(to_remove), int(num))

        if (str(distance) not in self.rolls.get()
                and self.white.get_pieces()[-1] >= 19
                and self.destination == 25
                and is_less
                and self.has_no_lower_points(self.white.get_pieces(), self.selected, to_remove)):
            try:
                self.white.move_piece(distance, self.selected, self.black, r)
                r.remove(str(to_remove))
                r = str(r)
                r = r[1:-1]
                r = r.replace(',', '')
                r = r.replace("'", '')
                self.rolls.set(r)
            except:
                self.title.set("Unvalid move!")

        elif str(distance) not in self.rolls.get():
            self.title.set("You can't move your piece there!")
            return
        else:
            try:
                if str(distance) in self.rolls.get() or self.white.get_pieces()[-1] <=6:
                    self.white.move_piece(distance, self.selected, self.black, r)
            except Exception as e:
                self.title.set(e)
                return
            r = self.rolls.get()
            if len(r) > 1:
                r = r.split()
                if str(distance) in r:
                    r.remove(str(distance))
                    r = str(r)
                    r = r[1:-1]
                    r = r.replace(',', '')
                    r = r.replace("'", '')
                    self.rolls.set(r)
                elif self.white.get_pieces()[-1] <= 6:  #was <=6
                    r.remove(str(r[random.randint(0,len(r) - 1)]))
                    r = str(r)
                    r = r[1:-1]
                    r = r.replace(',', '')
                    r = r.replace("'", '')
                    self.rolls.set(r)

            else:
                if str(distance) in self.rolls.get() or self.white.get_pieces()[-1] <= 6:
                    self.rolls.set('')
            if self.white.win():
                self.rolls.set('')
        if self.white.win():
            self.title.set('You won! Congratulations!')
            self.rollButton.config(state=DISABLED)
            self.endButton.config(state=DISABLED)
            self._canvas.unbind('<Button-1>')
            self._canvas.unbind('<Button-3>')
            return
        elif self.rolls.get() == '':
            self.black_turn()
        elif self.rolls.get() != '':
            self.title.set('Choose a piece to move')



    def goto(self, event):
        '''sets destination of piece based on where the user right-clicked'''
        x = event.x
        y = event.y
        x = x // TRI_WIDTH
        y = y // TRI_HEIGHT
        if y == 0:
            self.destination = 12 - x
        elif y == 1:
            self.destination = 0
        else:
            self.destination = 13 + x

    def end_turn(self):
        self.rolls.set('')
        self.time_remaining.set('')  # Clear timer display
        self.black_turn()

    def black_turn(self, set_timer=True):
        """Automates the black player's turn."""
        # Generate dice rolls for the black player
        computer_roll = roll()  # Dice roll
        print("computer roll: ", computer_roll)
        board = self.status_format()  # Get board status

        try:
            # Play a move using the black AI logic
            print("----------before move----------------")
            print("board:", board)
            print("computer pieses: ", self.black.get_pieces())
            print("Human pieces: ", self.white.get_pieces())
            move = self.black.play(board, computer_roll, "black")
            print("chosen moves: ", move)
            board = self.status_format()
            print("----------after move----------------")
            print("board:", board)
            print("computer pieses: ", self.black.get_pieces())
            print("Human pieces: ", self.white.get_pieces())

            # Check for a win
            if self.black.win():
                self.title.set('Black has won the game! Better luck next time.')
                self.rollButton.config(state=DISABLED)
                self.endButton.config(state=DISABLED)
                self._canvas.unbind('<Button-1>')
                self._canvas.unbind('<Button-3>')
                return

        except ValueError as e:
            # If no valid moves can be made, break out of the loop
            print(e)
            pass

        self.white.set_pieces(self.black.get_other_pieces())
        # After black's turn, enable the white player's turn
        self.title.set("It's your turn! Roll the dice!")
        self.rollButton.config(state=NORMAL)
        if set_timer:
            self.start_timer()  # Restart the timer for the next turn


if __name__ == '__main__':
    root = Tk()
    root.title('Backgammon')
    app = BackgammonGame(root)
    root.mainloop()