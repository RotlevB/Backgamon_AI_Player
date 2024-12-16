'''Creates a class for AI backgammon pieces
Created 2024
@author: Anni Ainesaz
'''
import random
from itertools import filterfalse


class AI_Player:
    def __init__(self, color = "black"):
        if color == "white":
            self._pieces = [1, 1,
                            12, 12, 12, 12, 12,
                            17, 17, 17,
                            19, 19, 19, 19, 19]
        else:
            self._pieces = [6, 6, 6, 6, 6,
                        8, 8, 8,
                        13, 13, 13, 13, 13,
                        24, 24]
        self.color = color

    def get_pieces(self):
        return self._pieces

    def set_pieces(self, list_of_pieces):
        self._pieces = list_of_pieces
        self.order()

    def get_other_pieces(self):
        return self.other_pieces

    def move_piece(self, distance, piece, r):  # piece is where to move from
        if distance <= 0:
            raise ValueError('Distance must be greater than 0')
        if ((self.color == "black" and piece != 25) or (self.color == "white" and piece != 0)) and self.capturedPiece():
            raise ValueError('You must move your captured piece first')
        if piece in self._pieces:
            idx = self._pieces.index(piece)
            if self.color == "black" and self.validMove(piece, piece - distance, r):
                self._pieces[idx] = piece - distance

            elif self.color == "white" and self.validMove(piece, piece + distance, r):
                self._pieces[idx] = piece + distance
            else:
                raise ValueError('That is an invalid place to move your piece')
        else:
            raise ValueError('The chosen piece is not valid')
        self.capture()
        self.order()

    def order(self):
        self._pieces.sort()

    def __str__(self):
        return 'Your pieces are at: ' + str(self._pieces)

    def capturedPiece(self):
        if ((self.color == "black") and (25 in self._pieces)) or ((self.color == "white") and (0 in self._pieces)):
            return True
        else:
            return False

    def capture(self):
        op = self.other_pieces[:]
        for piece in self._pieces:
            if piece in op:
                if self.color == "black":
                    op[op.index(piece)] = 0
                elif self.color == "white":
                    op[op.index(piece)] = 25
        self.other_pieces = op

    # def validMove(self, position):  # position is where to move
    #     res = 0
    #     # The following line was taken almost straight from stackoverflow (http://stackoverflow.com/questions/9542738/python-find-in-list)
    #     idx = [i for i, x in enumerate(self.other_pieces) if x == position]
    #     if len(idx) < 1:
    #         res += 1  # there is no other piece in position
    #     if self.color == "black" and position <= 0:
    #         if self._pieces[14] <= 6:  # all pieces at home
    #             res += 1
    #     elif self.color == "white" and position >= 25:
    #         if self._pieces[0] >= 19:  # all pieces at home
    #             res += 1
    #     elif 0 < position < 25:
    #         res += 1
    #     if res == 2:
    #         return True
    #     else:
    #         return False

    def validMove(self, makor, yaad, r, pieces=None):
        if pieces is None:
            pieces = self._pieces
        idx = [i for i, x in enumerate(self.other_pieces) if (x == yaad and x != 0 and x != 25)]
        no_oponent = len(idx) <= 1  # the oponent has max 1 pieces in yaad
        print(no_oponent)

        if str(yaad - makor) in r and 1 <= int(yaad) <= 24:
            return no_oponent

        elif self.color == "white" and yaad >= 25:  # the player wants to get piece out of home
            can_out = pieces[0] >= 19 # checks that all pieces in home
            if str(abs(yaad - makor)) not in r:
                relevant_roll = [x for x in r if int(yaad - makor) <= int(x)]
                if relevant_roll:
                    distance = min(relevant_roll)
                    # all pieces at home and there are no pieces between
                    pieces_between = [x for x in pieces if (25 - int(distance) <= x < int(makor))]
                    can_out = can_out and (not pieces_between)
            return can_out

        elif self.color == "black" and yaad <= 0:  # the player wants to get piece out of home
            can_out = pieces[14] <= 6 # checks that all pieces in home
            if str(abs(yaad - makor)) not in r:
                relevant_roll = [x for x in r if abs(yaad - makor) <= int(x)]
                if relevant_roll:
                    distance = min(relevant_roll)
                    # all pieces at home and there are no pieces between
                    pieces_between = [x for x in pieces if (int(makor) < x <= int(distance))]
                    can_out = can_out and (not pieces_between)
            return can_out

        return no_oponent

    def play(self, board, roll, color):
        """Get the board state, dice roll, and player color, and return the chosen move."""
        self.color = color
        self.roll = roll
        self._pieces = []  # Reset current player pieces
        self.other_pieces = []  # Reset opposing player pieces

        # Populate self._pieces and self.other_pieces based on board state
        # the first 24
        for i in range(len(board) - 4):
            if board[i] > 0:  # White pieces
                if color == "white":
                    self._pieces.extend([i + 1] * board[i])
                else:
                    self.other_pieces.extend([i + 1] * board[i])
            elif board[i] < 0:  # Black pieces
                if color == "black":
                    self._pieces.extend([i + 1] * abs(board[i]))
                else:
                    self.other_pieces.extend([i + 1] * abs(board[i]))

        # the last 4
        if self.color == "black":
            self._pieces.extend([0] * board[25])
            self._pieces.extend([25] * board[27])
            self.other_pieces.extend([25] * board[24])
            self.other_pieces.extend([0] * board[26])

        elif self.color == "white":
            self.other_pieces.extend([0] * board[25])
            self.other_pieces.extend([25] * board[27])
            self._pieces.extend([25] * board[24])
            self._pieces.extend([0] * board[26])

        print("before {} move: ".format(color))
        print("white pieces: ", self._pieces if self.color == "white" else self.other_pieces)
        print("black pieces: ", self._pieces if self.color == "black" else self.other_pieces)


        self.order()  # Ensure pieces are ordered
        self.other_pieces.sort()

        print("len:", len(self._pieces))
        whole_move = []

        while self.roll:
            # Generate all valid moves
            r = self.roll.pop()
            selected_move = self.select_move(r)
            whole_move.append(selected_move)

            if self.win() or self.lose():
                return
            # Apply the move
            if(selected_move):
                self.move_piece(abs(selected_move[1] - selected_move[0]), selected_move[0], roll)

        print("after {} move: ".format(color))
        print("white pieces: ", self._pieces if self.color == "white" else self.other_pieces)
        print("black pieces: ", self._pieces if self.color == "black" else self.other_pieces)
        return whole_move

    def generate_all_moves(self, r, pieces=None):
        if pieces is None:
            pieces = self._pieces
        self.all_moves = []

        for makor in set(pieces):

            if self.color == "black":
                yaad = makor - r
                if yaad < 0:
                    yaad = 0
                if self.validMove(makor, yaad, [r]):
                    self.all_moves.append([makor, yaad])
            elif self.color == "white":
                yaad = makor + r
                if yaad > 25:
                    yaad = 25
                if self.validMove(makor, yaad, [r]):
                    self.all_moves.append([makor, yaad])
        print("all moves: ", self.all_moves)
        return self.all_moves

    def select_move(self, r):
        return self.random_move(r)

    def random_move(self, r):

        valid_moves = self.generate_all_moves(r)
        if not valid_moves:
            raise ValueError("No valid moves available.")  # Skip turn if no moves

        # Pick a move randomly (can replace with strategic selection)
        selected_move = random.choice(valid_moves)
        return selected_move

    def win(self):

        if (self.color == "black" and self._pieces == [0] * 15) or (
                self.color == "white" and self._pieces == [25] * 15):
            # print("comp won!!!")
            return True
        else:
            # print("comp lost!!!")
            return False

    def lose(self):
        return ((self.color == "black" and self.other_pieces == [25] * 15)
                or (self.color == "white" and self.other_pieces == [0] * 15))







if __name__ == '__main__':
    # No tests here because all tests are in backgammon_white.py and almost all of it is the same
    a = AI_Player()