'''Creates a class for Human backgammon pieces
Created 2024
@author: Anni Ainesaz
'''

class Human_Player:
    def __init__(self):
        self._pieces = [1, 1,
                        12, 12, 12, 12, 12,
                        17, 17, 17,
                        19, 19, 19, 19, 19]

    def get_pieces(self):
        return self._pieces

    def set_pieces(self, list_of_pieces):
        self._pieces = list_of_pieces
        self.order()

    def move_piece(self, distance, piece, other, r):  # piece is where to move from
        if distance <= 0:
            raise ValueError('Distance must be greater than 0')
        if piece != 0 and self.capturedPiece():
            raise ValueError('You must move your captured piece first')
        if piece in self._pieces:
            idx = self._pieces.index(piece)
            if self.validMove(piece, piece + distance, other, r):
                self._pieces[idx] = piece + distance
            else:
                raise ValueError('That is an invalid place to move your piece')
        else:
            raise ValueError('The chosen piece is not valid')
        self.capture(other)
        self.order()

    def __str__(self):
        return 'Your pieces are at: ' + str(self._pieces)

    def order(self):
        self._pieces.sort()

    def capturedPiece(self):  # checks if I was eaten
        if 0 in self._pieces:
            return True
        else:
            return False

    def capture(self, other):  # I eat
        op = other.get_pieces()[:]
        for piece in self._pieces:
            if piece in op:
                op[op.index(piece)] = 25
        other.set_pieces(op)

    # def validMove(self, makor, yaad, other):
    #     res = 0
    #     #The following line was taken almost straight from stackoverflow (http://stackoverflow.com/questions/9542738/python-find-in-list)
    #     idx = [i for i,x in enumerate(other.get_pieces()) if (x == yaad and x != 0)]
    #     if len(idx) < 1:
    #         res += 1
    #     if yaad >= 25:
    #         distance = makor - yaad
    #         if self._pieces[0] >= 19 and not [x for x in self._pieces if  25 - distance <= x < makor]: #all pieces at home
    #             res += 1
    #     elif 0 < yaad < 25:
    #         res += 1
    #     if res == 2:
    #         return True
    #     else:
    #         return False

    def validMove(self, makor, yaad, other, r):

        idx = [i for i, x in enumerate(other.get_pieces()) if (x == yaad and x != 0 and x != 25)]
        no_oponent = len(idx) <= 1  # the oponent has max 1 pieces in yaad
        print("yaad - makor: ", str(yaad - makor))

        if str(yaad - makor) in r and 1 <= int(yaad) <= 24:
            return no_oponent

        elif yaad >= 25:  # the player wants to get piece out of home
            can_out = self._pieces[0] >= 19
            if str(abs(yaad - makor)) not in r:
                distance = max([x for x in r if int(x) <= int(makor)])
                # all pieces at home and there are no pieces between
                pieces_between = [x for x in self._pieces if (25 - int(distance) <= x < int(makor))]
                can_out = can_out and (not pieces_between)
            return can_out

        return no_oponent

    def win(self):
        return self._pieces == [25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25]


if __name__ == '__main__':
    player = Human_Player()
    p1 = Human_Player()
    p1.set_pieces([])
    assert player.get_pieces() == [1, 1, 12, 12, 12, 12, 12, 17, 17, 17, 19, 19, 19, 19, 19]
    player.move_piece(10, 1, p1)
    assert player.get_pieces() == [1, 11, 12, 12, 12, 12, 12, 17, 17, 17, 19, 19, 19, 19, 19]
    print(player)
    try:
        player.move_piece(0, 1, p1)
        print('Error with move_piece')
    except:
        print('move_piece working well with distance')
    try:
        player.move_piece(1, 20, p1)
        print('Error with move_piece')
    except:
        print('move_piece working well with pieces')
    player.set_pieces([0, 0, 0, 1, 0, 0, 0])
    assert player.get_pieces() == [0, 0, 0, 0, 0, 0, 1]
    try:
        player.move_piece(1, 1)
    except:
        print('capturedPiece is working well')
    player.move_piece(1, 0, p1)
    assert player.get_pieces() == [0, 0, 0, 0, 0, 1, 1]
    p1.set_pieces([1, 3, 3, 4, 5])
    player.capture(p1)
    assert p1.get_pieces() == [3, 3, 4, 5, 25]
    p1.set_pieces([2, 2])
    assert player.validMove(2, p1) == False
    assert player.validMove(3, p1) == True
    p1.set_pieces([5, 5])
    try:
        player.move_piece(4, 1, p1)
    except:
        print('validMove is working inside of move_piece')
    player.move_piece(4, 0, p1)
    assert player.get_pieces() == [0, 0, 0, 0, 1, 1, 4]
    player.set_pieces([25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25])
    assert player.win() == True

    print('All tests passed')