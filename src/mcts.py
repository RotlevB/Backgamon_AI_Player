import math
import time
from src.strategies import Strategy
import random
import numpy as np

class CompareAllMoves(Strategy):

    @staticmethod
    def get_difficulty():
        return "Hard"

    def assess_board(self, colour, myboard):
        pieces = myboard.get_pieces(colour)
        pieces_on_board = len(pieces)
        sum_distances = 0
        number_of_singles = 0
        number_occupied_spaces = 0
        sum_single_distance_away_from_home = 0
        sum_distance_away_from_home = 0
        sum_distances_to_endzone = 0
        hitting_opponent = 0
        secured_pieces = 0
        taken = len(myboard.get_taken_pieces(colour))
        for piece in pieces:
            sum_distances = sum_distances + piece.spaces_to_home()
            if piece.spaces_to_home() > 6:
                sum_distances_to_endzone += piece.spaces_to_home() - 6
        for location in range(1, 25):
            pieces = myboard.pieces_at(location)
            if len(pieces) != 0 and pieces[0].colour == colour:
                if len(pieces) == 1:
                    number_of_singles = number_of_singles + 1
                    sum_single_distance_away_from_home += 25 - pieces[0].spaces_to_home()
                elif len(pieces) > 1:
                    secured_pieces += len(pieces)
                    number_occupied_spaces = number_occupied_spaces + 1
                    sum_distance_away_from_home += (25 - pieces[0].spaces_to_home())* len(pieces)
            if len(pieces) != 0 and pieces[0].colour == colour.other():
                if len(pieces) == 1:
                    hitting_opponent = hitting_opponent + 1
        opponents_taken_pieces = len(myboard.get_taken_pieces(colour.other()))
        opponent_pieces = myboard.get_pieces(colour.other())
        sum_distances_opponent = 0
        for piece in opponent_pieces:
            sum_distances_opponent = sum_distances_opponent + piece.spaces_to_home()
        return {
            'number_occupied_spaces': number_occupied_spaces,
            'taken_pieces': taken,
            'opponents_taken_pieces': opponents_taken_pieces,
            'sum_distances': sum_distances,
            'sum_distances_opponent': sum_distances_opponent,
            'number_of_singles': number_of_singles,
            'sum_single_distance_away_from_home': sum_single_distance_away_from_home,
            'sum_distance_away_from_home': sum_distance_away_from_home,
            'pieces_on_board': pieces_on_board,
            'sum_distances_to_endzone': sum_distances_to_endzone,
            'hitting_opponent': hitting_opponent,
            'secured_pieces': secured_pieces,
            'finished_pieces': 15 - pieces_on_board - 2*taken,
        }

class Heuristic(CompareAllMoves):

    def evaluate_board(self, myboard, colour):

        board_stats = self.assess_board(colour, myboard)
        board_value = board_stats['opponents_taken_pieces'] + 3 * board_stats['finished_pieces'] -\
                      float(board_stats['sum_single_distance_away_from_home']) / 6 - \
                      board_stats['number_occupied_spaces'] + board_stats['opponents_taken_pieces'] - \
                      3 * board_stats['taken_pieces'] - float(board_stats['sum_distances_to_endzone']) / 6

        return board_value
    

class MCTSStrategy:

    def get_difficulty(self):
        return "Hard"  # This can be adjusted based on MCTS configuration
    
    def TreePolicy(self, v, player):                    # TreePolicy(v) -
        while not v.is_terminal:                # while v is non-terminal do:
            if not v.is_fully_expanded:         #   if v not fully expanded then:
                return v.expand()               #       return Expand(v)
            else:                               #   else:
                v = v.best_child(player)        #       v <- BestChild(v)
                                                #   end if
                                                # end while
        return v                                # return v

    def Backup(self, v, delta):                 # Backup(v, delta) -
        while v is not None:                    # while v is not null do:
            v.N += 1                            #   N(v) <- N(v) + 1
            v.Q += delta                        #   Q(v) <- Q(v) + delta(v,p)
            v = v.pa                            #   v <- Parent(v)
                                                # end while
        return 

    def move(self, board, player, dice_rolls, make_move, opponents_activity):

        # Start tracking time to ensure we do not exceed the time limit
        time_limit = board.getTheTimeLim()
        start_time = time.time()
        limit_buffered = time_limit - 0.1  # Small buffer to ensure we stay within the time limit

        # Create root node v0 woth state s0
        root = MCTSActionNode(board, dice_rolls, player)

        # Perform simulations until the time limit is reached

        # while within computational budget do:
        while time.time() - start_time < limit_buffered:
            v1 = self.TreePolicy(root, player)  # v1 <- TreePolicy(v0)
            delta = v1.simulate(player)  # delta <- DefaultPolicy(s0)
            self.Backup(v1, delta)

        # Choose the best move based on visit counts or win rates
        best_child = root.best_child(player)
        best_move =  best_child.move_sequence
        for move in best_move:
            make_move(move[0].location, move[1])

class MCTSNode:

    def __init__(self, board, player, parent=None, move_sequence=None):
        self.board = board  # Copy of the game board state
        self.player = player  # Player to move
        self.pa = parent  # Parent node
        self.children = {}  # child nodes
        self.move_sequence = move_sequence or []  # Sequence of moves leading to this node
        self.N = 0  # Number of times this node has been visited
        self.Q = 0  # Total value from this node's simulations
        self.c = 0.5  # Exploration factor for UCB
        self.is_terminal = board.has_game_ended()
        self.is_fully_expanded = False
    
    def evaluate_board(self, board, colour):
        evaluator = Heuristic()
        return evaluator.evaluate_board(board, colour)
    
    def UCB(self, sign):

        return ((sign * self.Q) / self.N) + self.c * ((math.log(self.pa.N) / self.N) ** 0.5)

    def simulate(self, player):
        return self.evaluate_board(self.board, player)

    def best_child(self, player):
        sign = 1 if self.player == player else -1
        return max(self.children.values(), key=lambda child: child.UCB(sign))

class MCTSStateNode(MCTSNode):
    """
    Represents a node in the MCTS tree.
    Each node corresponds to a game state and holds statistics for UCB1 calculation.
    """

    def __init__(self, board,  player, parent=None, move_sequence=None):
        super().__init__(board, player, parent, move_sequence)

    @staticmethod
    def generate_dice_rolls():
        dice1 = random.randint(1, 6)
        dice2 = random.randint(1, 6)
        if dice1 == dice2:
            return [dice1] * 4
        elif dice1 < dice2:
            return [dice1, dice2]
        else:
            return [dice2, dice1]

    def expand(self):
        dice_rolls = self.generate_dice_rolls()
        dice_rolls_key = tuple(dice_rolls)
        if dice_rolls_key not in self.children:
            self.children[dice_rolls_key] = MCTSActionNode(self.board, dice_rolls, self.player.other(), self, self.move_sequence)
        return self.children[dice_rolls_key]

class MCTSActionNode(MCTSNode):
    def __init__(self, board, dice_rolls, player, parent=None, move_sequence=None):
        super().__init__(board, player, parent, move_sequence)
        self.dice_rolls = dice_rolls  # Remaining dice rolls for this node
        self.possible_moves = self.generate_possible_moves()  # Lazy move sequence generator
        self.next_move = self.next_unplayed_move()  # Next possible move sequence

    @staticmethod
    def apply_move(board, move):
        piece, die_roll = move
        if board.is_move_possible(piece, die_roll):
            board_copy = board.create_copy()
            new_piece = board_copy.get_piece_at(piece.location)
            board_copy.move_piece(new_piece, die_roll)
            return board_copy
        return None

    @staticmethod
    def get_possible_moves(board, colour, die_roll):
        """
        Get all valid moves for a specific die roll.
        """
        pieces = board.get_taken_pieces(colour)
        if not pieces:
            pieces = board.get_pieces(colour)
            pieces = sorted(pieces, key=lambda piece: piece.spaces_to_home(), reverse=True)

        for piece in pieces:
            if board.is_move_possible(piece, die_roll):
                yield (piece, die_roll)

    def generate_possible_moves(self, board=None, dice_rolls=None, move_sequence=None):
        """
        Recursively generate sequences of moves lazily, accounting for all permutations of dice usage.
        """
        board = board or self.board
        dice_rolls = dice_rolls or self.dice_rolls
        move_sequence = move_sequence or []

        if not dice_rolls or board.has_game_ended():  # Base case: no more dice to use
            yield move_sequence
            return

        # Generate all permutations of the dice rolls
        if (dice_rolls[0] != dice_rolls[1]):
            yield from self._generate_moves_for_dice_order(board, [dice_rolls[1], dice_rolls[0]], move_sequence)
        
        yield from self._generate_moves_for_dice_order(board, dice_rolls, move_sequence)

    def _generate_moves_for_dice_order(self, board, dice_order, move_sequence):
        """
        Helper method to recursively generate moves for a specific dice order.
        """
        if not dice_order or board.has_game_ended():  # Base case: no more dice in this permutation
            yield move_sequence
            return

        current_die = dice_order[0]
        remaining_dice = dice_order[1:]

        # Generate moves for the current die
        for move in self.get_possible_moves(board, self.player, current_die):
            # Apply the move to get the new board state
            board_copy = self.apply_move(board, move)

            # Recursively generate moves for the remaining dice
            yield from self._generate_moves_for_dice_order(
                board=board_copy,
                dice_order=remaining_dice,
                move_sequence=move_sequence + [move],
            )
        
    def get_next_move(self):
        """
        Returns the next possible move sequence from the generator.
        """
        try:
            # Get the next sequence of moves from the generator
            return next(self.possible_moves)
        except StopIteration:
            # If there are no more sequences, return None or any sentinel value
            return None
        
    def next_unplayed_move(self):
        next_move_sequence = self.get_next_move()
        if next_move_sequence is None:
            return None  # No more moves to expand
        
        # Convert the move sequence to a tuple to use as a dictionary key
        move_sequence_key = tuple(next_move_sequence)

        # Check if the move sequence already exists in the children
        while move_sequence_key in self.children:
            next_move_sequence = self.get_next_move()
            move_sequence_key = tuple(next_move_sequence)

        return next_move_sequence

    def expand(self):
        """
        Expands the current node by generating all possible child nodes.
        """

        next_move_sequence = self.next_move

        if next_move_sequence is None:
            next_move_sequence = []

        move_sequence_key = tuple(next_move_sequence)
        
        # Update the next move sequence for the next expansion
        if next_move_sequence is []:
            self.next_move = None
            self.is_fully_expanded = True
            new_board = self.board.create_copy()
            new_child = MCTSStateNode(new_board, self.player, self, next_move_sequence)

        else:
            self.next_move = self.next_unplayed_move()
            if self.next_move is None:
                self.is_fully_expanded = True

            new_board = self.board.create_copy()
            for move in next_move_sequence:
                new_board = self.apply_move(new_board, move)
            new_child = MCTSStateNode(new_board, self.player, self, next_move_sequence)


        self.children[move_sequence_key] = new_child

        return self.children[move_sequence_key]
            