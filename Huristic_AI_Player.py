'''
created by: Benjamin Rotlevy
last update: 16.12.2024
'''
import itertools

from AI_Player import *

class Huristic_AI_Player(AI_Player):
    def __init__(self, color="black"):
        super().__init__(color)

    def heuristic_function(self, color, _pieces, other_pieces):
        """
        Evaluate the board state and return a heuristic score.

        Args:
            color (str): The player's color ("white" or "black").
            _pieces (list): A 15-length list of the player's piece positions.
            other_pieces (list): A 15-length list of the opponent's piece positions.

        Returns:
            float: A heuristic score for the current board state.
        """
        # Define scoring weights
        WEIGHTS = {
            "piece_safety": 10,
            "board_control": 5,
            "race_progress": 3,
            "captured_pieces": -15,
            "escaped_pieces": 10,
        }

        # Identify home and escape indices
        home_index = 0 if color == "black" else 25
        escape_index = 25 if color == "black" else 0

        # Initialize scores
        piece_safety_score = 0
        board_control_score = 0
        race_progress_score = 0
        captured_pieces_score = 0
        escaped_pieces_score = 0

        # Calculate piece safety and board control
        position_counts = [0] * 26
        for piece in _pieces:
            position_counts[piece] += 1

        for idx, count in enumerate(position_counts):
            if count == 1:  # Blot
                piece_safety_score -= 1
            elif count > 1:  # Safe point
                piece_safety_score += 1
                board_control_score += 1  # More checkers = stronger control

        # Calculate race progress
        for piece in _pieces:
            if piece not in [home_index, escape_index]:  # Not captured or escaped
                race_progress_score += (escape_index - piece if color == "white" else piece - home_index)

        # Calculate penalties for captured pieces
        captured_pieces_score = -_pieces.count(home_index) * WEIGHTS["captured_pieces"]

        # Calculate rewards for escaped pieces
        escaped_pieces_score = _pieces.count(escape_index) * WEIGHTS["escaped_pieces"]

        # Combine all scores
        total_score = (
                WEIGHTS["piece_safety"] * piece_safety_score +
                WEIGHTS["board_control"] * board_control_score +
                WEIGHTS["race_progress"] * race_progress_score +
                captured_pieces_score +
                escaped_pieces_score
        )

        return total_score

    def simulate_move(self, move, board=None):
        """
        Simulate a move and return the resulting board state.

        Args:
            move (tuple): A tuple representing the chosen move (piece index, move distance).

        Returns:
            list: A 28-length list representing the resulting board state.
        """
        # Initialize variables
        if board is None:
            board = self._pieces.copy()
        new_board = board
        source, destination = move

        # Move the piece
        index = new_board.index(source)
        new_board[index] = destination

        return new_board

    def simulate_moves(self, moves):
        """
        Simulate a sequence of moves and return the resulting board state.

        Args:
            moves (list): A list of tuples representing the chosen moves (piece index, move distance).

        Returns:
            list: A 28-length list representing the resulting board state.
        """
        new_board = None
        for move in moves:
            new_board = self.simulate_move(move, new_board)
        return new_board

    def minimax_with_probability(self, depth, maximizing_player, board, dice_rolls, alpha, beta, probability=1.0):
        """
        Minimax function with dice roll probabilities.

        Args:
            depth (int): Maximum depth of the game tree to explore.
            maximizing_player (bool): True if it's the maximizing player's turn.
            board (list): Current board state.
            dice_rolls (list): Current dice rolls to evaluate.
            alpha (float): Alpha value for alpha-beta pruning.
            beta (float): Beta value for alpha-beta pruning.
            probability (float): Cumulative probability of reaching this state.

        Returns:
            float: The evaluation score of the board.
        """
        if depth == 0 or probability < 0.01:
            # Base case: Return heuristic score at leaf nodes
            return self.heuristic_function(self.color, self._pieces, self.other_pieces)

        all_possible_moves = self.generate_moves_for_two_dice(dice_rolls)
        if not all_possible_moves:
            # No valid moves: End this branch
            return self.heuristic_function(self.color, self._pieces, self.other_pieces)

        best_value = float('-inf') if maximizing_player else float('inf')

        for moves in all_possible_moves:
            new_board = self.simulate_moves(moves)
            next_dice_rolls = list(itertools.product(range(1, 7), repeat=2))  # All dice combinations
            for dice in next_dice_rolls:
                prob = 1 / 36 if dice[0] == dice[1] else 2 / 36  # Probability for doubles and non-doubles
                score = self.minimax_with_probability(
                    depth - 1,
                    not maximizing_player,
                    new_board,
                    list(dice),
                    alpha,
                    beta,
                    probability * prob
                )

                if maximizing_player:
                    best_value = max(best_value, score)
                    alpha = max(alpha, best_value)
                else:
                    best_value = min(best_value, score)
                    beta = min(beta, best_value)

                # Alpha-Beta Pruning
                if beta <= alpha:
                    break

        return best_value

    def huristic_depth_Move(self, dice_rolls, depth=2):
        """
        Select the best move using Minimax with dice probabilities and a specified depth.

        Args:
            dice_rolls (list): The initial dice rolls for the turn.
            depth (int): The depth of the Minimax search tree.

        Returns:
            list: The best sequence of moves.
        """
        best_moves = None
        best_score = float('-inf')
        all_moves = self.generate_moves_for_two_dice(dice_rolls)

        for moves in all_moves:
            simulated_board = self.simulate_moves(moves)
            next_dice_rolls = list(itertools.product(range(1, 7), repeat=2))  # All dice combinations
            score = 0
            for dice in next_dice_rolls:
                prob = 1 / 36 if dice[0] == dice[1] else 2 / 36
                score += prob * self.minimax_with_probability(
                    depth=depth,
                    maximizing_player=False,
                    board=simulated_board,
                    dice_rolls=dice,
                    alpha=float('-inf'),
                    beta=float('inf'),
                    probability=prob
                )

            if score > best_score:
                best_score = score
                best_moves = moves

        return best_moves

    def Huristic_Move(self, dice_rolls):
        """
        Get the current board state and dice roll, and return the chosen move.

        Args:
            board (list): A 28-length list representing the board state.
            roll (list): A 2-length list representing the dice roll.

        Returns:
            tuple: A tuple representing the chosen move (piece index, move distance).
        """
        # Initialize variables
        best_moves = None
        best_score = float("-inf")

        # Generate all possible moves
        all_moves = self.generate_moves_for_two_dice(dice_rolls)
        print("all_moves: ", all_moves)

        # Evaluate each move using the heuristic function
        for moves in all_moves:
            print("pieces: ", self._pieces)
            print("moves: ", moves)
            new_board = self.simulate_moves(moves)
            print("new_board: ", new_board)
            try:
                score = self.heuristic_function(self.color, new_board, self.other_pieces)
                print("score: ", score)
            except ValueError:
                pass  # Skip turn if no valid moves
            if score > best_score:
                best_moves = moves
                best_score = score

        print("best_moves: ", best_moves)
        return best_moves

    def generate_moves_for_two_dice(self, dice_rolls):
        """
        Generate all possible moves using two dice rolls.

        Args:
            dice_rolls (list): A list of two dice rolls, e.g., [3, 4].

        Returns:
            list: A list of all possible moves using the dice rolls.
        """
        # Generate moves for each die separately
        first_die_moves = self.generate_all_moves(dice_rolls[0])
        second_die_moves = self.generate_all_moves(dice_rolls[1])

        # Combine all move possibilities
        all_moves = []

        # Moves using the first die followed by the second die
        for first_move in first_die_moves:
            temp_pieces = self.simulate_move(first_move)  # Update board state after first move
            second_moves = self.generate_all_moves(dice_rolls[1], temp_pieces)
            if not second_moves:  # Skip if no valid moves
                all_moves.append([first_move])
            for second_move in second_moves:
                all_moves.append([first_move, second_move])

        # Moves using the second die followed by the first die
        for second_move in second_die_moves:
            temp_pieces = self.simulate_move(second_move)
            first_moves = self.generate_all_moves(dice_rolls[0], temp_pieces)
            if not first_moves:
                all_moves.append([second_move])
            for first_move in first_moves:
                all_moves.append([second_move, first_move])

        return all_moves

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
        whole_move = self.huristic_depth_Move(self.roll)
        print("whole_move: ", whole_move)
        if whole_move:
            for move in whole_move:
                if move:
                    self.move_piece(abs(move[1] - move[0]), move[0], roll)


        print("after {} move: ".format(color))
        print("white pieces: ", self._pieces if self.color == "white" else self.other_pieces)
        print("black pieces: ", self._pieces if self.color == "black" else self.other_pieces)
        return whole_move



    def select_move(self, r):
        return self.Huristic_Move(r)