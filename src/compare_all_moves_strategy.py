from itertools import permutations

from src.strategies import Strategy
from src.piece import Piece


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
        sum_distances_to_endzone = 0
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
                    number_occupied_spaces = number_occupied_spaces + 1
        opponents_taken_pieces = len(myboard.get_taken_pieces(colour.other()))
        opponent_pieces = myboard.get_pieces(colour.other())
        sum_distances_opponent = 0
        for piece in opponent_pieces:
            sum_distances_opponent = sum_distances_opponent + piece.spaces_to_home()
        return {
            'number_occupied_spaces': number_occupied_spaces,
            'opponents_taken_pieces': opponents_taken_pieces,
            'sum_distances': sum_distances,
            'sum_distances_opponent': sum_distances_opponent,
            'number_of_singles': number_of_singles,
            'sum_single_distance_away_from_home': sum_single_distance_away_from_home,
            'pieces_on_board': pieces_on_board,
            'sum_distances_to_endzone': sum_distances_to_endzone,
        }

    def move(self, board, colour, dice_roll, make_move, opponents_activity):

        result = self.move_recursively(board, colour, dice_roll)
        not_a_double = len(dice_roll) == 2
        if not_a_double:
            new_dice_roll = dice_roll.copy()
            new_dice_roll.reverse()
            result_swapped = self.move_recursively(board, colour,
                                                   dice_rolls=new_dice_roll)
            if result_swapped['best_value'] < result['best_value'] and \
                    len(result_swapped['best_moves']) >= len(result['best_moves']):
                result = result_swapped

        if len(result['best_moves']) != 0:
            for move in result['best_moves']:
                make_move(move['piece_at'], move['die_roll'])

    def move_recursively(self, board, colour, dice_rolls):
        best_board_value = float('inf')
        best_pieces_to_move = []

        pieces_to_try = [x.location for x in board.get_pieces(colour)]
        pieces_to_try = list(set(pieces_to_try))

        valid_pieces = []
        for piece_location in pieces_to_try:
            valid_pieces.append(board.get_piece_at(piece_location))
        valid_pieces.sort(key=Piece.spaces_to_home, reverse=True)

        dice_rolls_left = dice_rolls.copy()
        die_roll = dice_rolls_left.pop(0)

        for piece in valid_pieces:
            if board.is_move_possible(piece, die_roll):
                board_copy = board.create_copy()
                new_piece = board_copy.get_piece_at(piece.location)
                board_copy.move_piece(new_piece, die_roll)
                if len(dice_rolls_left) > 0:
                    result = self.move_recursively(board_copy, colour, dice_rolls_left)
                    if len(result['best_moves']) == 0:
                        # we have done the best we can do
                        board_value = self.evaluate_board(board_copy, colour)
                        if board_value < best_board_value and len(best_pieces_to_move) < 2:
                            best_board_value = board_value
                            best_pieces_to_move = [{'die_roll': die_roll, 'piece_at': piece.location}]
                    elif result['best_value'] < best_board_value:
                        new_best_moves_length = len(result['best_moves']) + 1
                        if new_best_moves_length >= len(best_pieces_to_move):
                            best_board_value = result['best_value']
                            move = {'die_roll': die_roll, 'piece_at': piece.location}
                            best_pieces_to_move = [move] + result['best_moves']
                else:
                    board_value = self.evaluate_board(board_copy, colour)
                    if board_value < best_board_value and len(best_pieces_to_move) < 2:
                        best_board_value = board_value
                        best_pieces_to_move = [{'die_roll': die_roll, 'piece_at': piece.location}]

        return {'best_value': best_board_value,
                'best_moves': best_pieces_to_move}


class CompareAllMovesSimple(CompareAllMoves):

    def evaluate_board(self, myboard, colour):
        board_stats = self.assess_board(colour, myboard)

        board_value = board_stats['sum_distances'] + 2 * board_stats['number_of_singles'] - \
                      board_stats['number_occupied_spaces'] - board_stats['opponents_taken_pieces']
        return board_value


class CompareAllMovesWeightingDistance(CompareAllMoves):

    def evaluate_board(self, myboard, colour):
        board_stats = self.assess_board(colour, myboard)

        board_value = board_stats['sum_distances'] - float(board_stats['sum_distances_opponent'])/3 + \
                      2 * board_stats['number_of_singles'] - \
                      board_stats['number_occupied_spaces'] - board_stats['opponents_taken_pieces']
        return board_value


class CompareAllMovesWeightingDistanceAndSingles(CompareAllMoves):

    def evaluate_board(self, myboard, colour):
        board_stats = self.assess_board(colour, myboard)

        board_value = board_stats['sum_distances'] - float(board_stats['sum_distances_opponent'])/3 + \
                      float(board_stats['sum_single_distance_away_from_home'])/6 - \
                      board_stats['number_occupied_spaces'] - board_stats['opponents_taken_pieces']
        return board_value


class CompareAllMovesWeightingDistanceAndSinglesWithEndGame(CompareAllMoves):

    def evaluate_board(self, myboard, colour):
        board_stats = self.assess_board(colour, myboard)

        board_value = board_stats['sum_distances'] - float(board_stats['sum_distances_opponent']) / 3 + \
                      float(board_stats['sum_single_distance_away_from_home']) / 6 - \
                      board_stats['number_occupied_spaces'] - board_stats['opponents_taken_pieces'] + \
                      3 * board_stats['pieces_on_board']

        return board_value


class CompareAllMovesWeightingDistanceAndSinglesWithEndGame2(CompareAllMoves):

    def evaluate_board(self, myboard, colour):
        board_stats = self.assess_board(colour, myboard)

        board_value = board_stats['sum_distances'] - float(board_stats['sum_distances_opponent']) / 3 + \
                      float(board_stats['sum_single_distance_away_from_home']) / 6 - \
                      board_stats['number_occupied_spaces'] - board_stats['opponents_taken_pieces'] + \
                      3 * board_stats['pieces_on_board'] + float(board_stats['sum_distances_to_endzone']) / 6

        return board_value

class HuristicCompareAllMoves(CompareAllMoves):
    def heuristic_function(self, state):
        """
        Evaluate the board state and return a heuristic score.

        Args:
            state (dict): A dictionary containing game state metrics.

        Returns:
            float: A heuristic score for the current board state.
        """
        # Extract values from the state object
        number_occupied_spaces = state['number_occupied_spaces']
        opponents_taken_pieces = state['opponents_taken_pieces']
        sum_distances = state['sum_distances']
        sum_distances_opponent = state['sum_distances_opponent']
        number_of_singles = state['number_of_singles']
        sum_single_distance_away_from_home = state['sum_single_distance_away_from_home']
        pieces_on_board = state['pieces_on_board']
        sum_distances_to_endzone = state['sum_distances_to_endzone']

        # Define weights for each attribute
        weights = {
            'number_occupied_spaces': 1.5,
            'opponents_taken_pieces': 2.0,
            'sum_distances': -0.5,  # Closer to home is better, so penalize higher distances
            'sum_distances_opponent': 0.5,  # Favor opponent being far from home
            'number_of_singles': -1.0,  # Penalize singles
            'sum_single_distance_away_from_home': -0.3,  # Penalize singles farther from home
            'pieces_on_board': 1.0,  # Favor having more pieces on the board
            'sum_distances_to_endzone': -0.7,  # Prefer pieces closer to the endzone
        }

        # Calculate heuristic score
        score = (
                weights['number_occupied_spaces'] * number_occupied_spaces +
                weights['opponents_taken_pieces'] * opponents_taken_pieces +
                weights['sum_distances'] * sum_distances +
                weights['sum_distances_opponent'] * sum_distances_opponent +
                weights['number_of_singles'] * number_of_singles +
                weights['sum_single_distance_away_from_home'] * sum_single_distance_away_from_home +
                weights['pieces_on_board'] * pieces_on_board +
                weights['sum_distances_to_endzone'] * sum_distances_to_endzone
        )

        return score * -1
    def evaluate_board(self, myboard, colour):
        board_stats = self.assess_board(colour, myboard)
        return self.heuristic_function(board_stats)

class MinimaxPlayer(HuristicCompareAllMoves):
    def __init__(self, depth=2):
        super().__init__()
        self.max_depth = depth

    def minimax(self, board, colour, depth, dice_rolls):
        """
        Minimax algorithm to evaluate the best sequence of moves based on dice rolls.

        Args:
            board (Board): Current board state.
            colour (Colour): Player's colour.
            depth (int): Current depth in the tree.
            dice_rolls (list): Sequence of dice rolls to simulate.

        Returns:
            tuple: (Best score, list of best moves)
        """
        if depth == 0 or board.has_game_ended():
            state = self.assess_board(colour, board)
            score = self.heuristic_function(state)
            return score, []

        best_score = float('-inf') if depth % 2 == 1 else float('inf')
        best_moves = []

        all_permutations = list(permutations(dice_rolls)) if len(dice_rolls) == 2 else [dice_rolls]
        for permutation in all_permutations:
            score, moves = self.evaluate_permutation(board, colour, depth, list(permutation))
            if (depth % 2 == 1 and score > best_score) or (depth % 2 == 0 and score < best_score):
                best_score = score
                best_moves = moves

        return best_score, best_moves

    def evaluate_permutation(self, board, colour, depth, roll_sequence):
        """
        Evaluate a specific sequence of dice rolls recursively.

        Args:
            board (Board): Current board state.
            colour (Colour): Player's colour.
            depth (int): Current depth in the tree.
            roll_sequence (list): Sequence of dice rolls to simulate.

        Returns:
            tuple: (Evaluation score, list of moves)
        """
        if not roll_sequence:
            return self.minimax(board, colour.other(), depth - 1, [])

        current_roll = roll_sequence[0]
        remaining_rolls = roll_sequence[1:]

        best_score = float('-inf') if depth % 2 == 1 else float('inf')
        best_moves = []

        for piece in board.get_pieces(colour):
            if not board.is_move_possible(piece, current_roll):
                continue

            # Create a copy of the board
            board_copy = board.create_copy()

            # Resolve the new piece on the copied board
            new_piece = board_copy.get_piece_at(piece.location)
            if new_piece is None:
                continue  # Skip invalid moves

            # Perform the move on the copied board
            board_copy.move_piece(new_piece, current_roll)

            # Recursively evaluate the next roll
            score, moves = self.evaluate_permutation(board_copy, colour, depth, remaining_rolls)

            # Track the best score and moves
            move = {'piece': piece, 'dice_roll': current_roll}
            if (depth % 2 == 1 and score > best_score) or (depth % 2 == 0 and score < best_score):
                best_score = score
                best_moves = [move] + moves

        return best_score, best_moves

    def move(self, board, colour, dice_rolls, make_move, opponents_activity):
        """
        Decide and execute the best move using the Minimax algorithm.

        Args:
            board (Board): Current board state.
            colour (Colour): Player's colour.
            dice_rolls (list): Dice rolls available for the turn.
            make_move (callable): Function to execute a move.
            opponents_activity (dict): Opponent's activity data.
        """
        best_score, best_moves = self.minimax(board, colour, self.max_depth, dice_rolls)

        for move in best_moves:
            piece = board.get_piece_at(move['piece'].location)
            if piece is None or not board.is_move_possible(piece, move['dice_roll']):
                continue
            try:
                make_move(move['piece'].location, move['dice_roll'])
            except Exception as e:
                print(f"Error executing move: {e}")
