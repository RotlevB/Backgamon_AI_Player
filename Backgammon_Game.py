'''Creates a class for Human backgammon pieces
Created 2024
@author: Anni Ainesaz
'''

from AI_Player import *
from Human_Player import *
import random
import time




def roll():
    r = [random.randint(1,6), random.randint(1,6)]
    if r[0] == r[1]:
        r = [r[0], r[0], r[0], r[0]]
    return r

if __name__ == '__main__':
    white = Backgammon_White()
    black = Backgammon_Black()
    while not black.win():
        #white turn
        white_roll = roll()
        while white_roll != [] and not white.win():
            print('Your roll is: ' + str(white_roll))
            print('Your pieces are at: ' + str(white.get_pieces()))
            piece = int(input('Which piece would you like to move: '))
            distance = -1
            while distance not in white_roll:
                distance = int(input('How far would you like to move the piece? (Make sure your answer is one of your rolls) '))
            white.move_piece(distance, piece, black)
            white_roll.remove(distance)

        print('Your pieces are now at: ' + str(white.get_pieces()))

        if white.win():
            break

        #black turn
        black_roll = roll()
        while black_roll != [] and not black.win():
            time.sleep(3) #Got this function from Stack Overflow
            print("My roll is: " + str(black_roll))
            time.sleep(3)
            print("My pieces are at: " + str(black.get_pieces()))
            roll_idx = random.randint(0, len(black_roll) - 1)
            roll_to_use = black_roll[roll_idx]
            piece_to_use = 0
            invalid = []
            while True:
                try:
                    while (not piece_to_use) and not(piece_to_use in invalid):
                        bpcopy = black.get_pieces()[:]
                        piece_to_use = bpcopy[random.randint(0,15)]
                    black.move_piece(roll_to_use, piece_to_use, white)
                    break
                except:
                    invalid.append(piece_to_use)
            time.sleep(3)
            print('I moved the piece at ' + str(piece_to_use) + ' to the position ' + str(piece_to_use - roll_to_use))
            black_roll.remove(roll_to_use)

    if black.win():
        print('Too bad for you! I won this time.')
    else:
        print('Good job! You managed to beat me this game.')