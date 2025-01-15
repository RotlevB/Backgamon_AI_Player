from src.strategies import Strategy
import time

class MinimaxNode:
	def __init__(self, board, colour, dice_rolls, depth=0, max_depth=2, move=None, parent=None, mode="max"):
		self.board = board
		self.colour = colour
		self.dice_rolls = dice_rolls
		self.depth = depth
		self.max_depth = max_depth
		self.parent = parent
		self.mode = mode
		self.children = []
		self.best_score = float('-inf') if mode == "max" else float('inf')
		self.move = move
	
	def assess_board(self, colour, myboard):
		pieces = myboard.get_pieces(colour)
		pieces_on_board = len(pieces)
		sum_distances = 0
		number_of_singles = 0
		number_occupied_spaces = 0
		sum_single_distance_away_from_home = 0
		sum_distances_to_endzone = 0
		pieces_at_home = 0
		for piece in pieces:
			sum_distances = sum_distances + piece.spaces_to_home()
			if piece.spaces_to_home() > 6:
				sum_distances_to_endzone += piece.spaces_to_home() - 6
			else:
				pieces_at_home += 1
		for location in range(1, 25):
			pieces = myboard.pieces_at(location)
			if len(pieces) != 0 and pieces[0].colour == colour:
				if len(pieces) == 1:
					number_of_singles = number_of_singles + 1
					sum_single_distance_away_from_home += 25 - pieces[0].spaces_to_home()
				elif len(pieces) > 1:
					number_occupied_spaces = number_occupied_spaces + 1
		opponents_taken_pieces = len(myboard.get_taken_pieces(colour.other()))
		taken_pieces = len(myboard.get_taken_pieces(colour))
		opponent_pieces = myboard.get_pieces(colour.other())
		sum_distances_opponent = 0
		for piece in opponent_pieces:
			sum_distances_opponent = sum_distances_opponent + piece.spaces_to_home()
		return {
			'number_occupied_spaces': number_occupied_spaces,
			'opponents_taken_pieces': opponents_taken_pieces,
			'taken_pieces': taken_pieces,
			'sum_distances': sum_distances,
			'sum_distances_opponent': sum_distances_opponent,
			'number_of_singles': number_of_singles,
			'sum_single_distance_away_from_home': sum_single_distance_away_from_home,
			'pieces_on_board': pieces_on_board,
			'sum_distances_to_endzone': sum_distances_to_endzone,
			'pieces_at_home': pieces_at_home,
		}

	def heuristic_function(self, state_stats):
		"""
		Evaluate the board state_stats and return a heuristic score.

		Args:
			state_stats (dict): A dictionary containing game state_stats metrics.

		Returns:
			float: A heuristic score for the current board state.
		"""
		if self.board.has_game_ended():
			if self.board.who_won() == self.colour:
				return 1000
			else:
				return -1000
		# Extract values from the state object
		number_occupied_spaces = state_stats['number_occupied_spaces']
		opponents_taken_pieces = state_stats['opponents_taken_pieces']
		sum_distances = state_stats['sum_distances']
		sum_distances_opponent = state_stats['sum_distances_opponent']
		number_of_singles = state_stats['number_of_singles']
		sum_single_distance_away_from_home = state_stats['sum_single_distance_away_from_home']
		pieces_on_board = state_stats['pieces_on_board']
		sum_distances_to_endzone = state_stats['sum_distances_to_endzone']
		finished_pieces = 15 - pieces_on_board - state_stats['taken_pieces']
		pieces_home = state_stats['pieces_at_home']
		pieces_out_of_home = pieces_on_board - pieces_home - state_stats['taken_pieces']
		average_distance_to_home = sum_distances_to_endzone / max(pieces_out_of_home, 1) if pieces_out_of_home > 0 else 0

		# Phase logic
		if pieces_out_of_home < 3 and average_distance_to_home < 12:
			phase = 'late'
		elif average_distance_to_home < 18 or opponent_pieces_home >= 3:
			phase = 'mid'
		else:
			phase = 'early'

		
		# Define weights for each phase
		phase_weights = {
			'early': {
				'number_occupied_spaces': 1.5,
				'opponents_taken_pieces': 2.0,
				'sum_distances': -0.8,
				'sum_distances_opponent': 0.5,
				'number_of_singles': -2.0,
				'pieces_on_board': 1.2,
				'sum_distances_to_endzone': -0.5,
				'sum_single_distance_away_from_home': -0.4,  # Heavily penalize singles far from home
				'finished_pieces': 2.0,
			},
			'mid': {
				'number_occupied_spaces': 1.5,
				'opponents_taken_pieces': 2.0,
				'sum_distances': -0.6,
				'sum_distances_opponent': 0.8,
				'number_of_singles': -3.5,
				'pieces_on_board': 1.0,
				'sum_distances_to_endzone': -0.9,
				'sum_single_distance_away_from_home': -0.3,  # Heavily penalize singles far from home
				'finished_pieces': 4.0,
			},
			'late': {
				'number_occupied_spaces': 0.5,
				'opponents_taken_pieces': 1.5,
				'sum_distances': -0.3,
				'sum_distances_opponent': 0.2,
				'number_of_singles': -1.0,
				'pieces_on_board': 1.5,
				'sum_distances_to_endzone': -1.0,
				'sum_single_distance_away_from_home': -0.1,  # Heavily penalize singles far from home
				'finished_pieces': 8.0,
			}
		}

		# Use weights for the current phase
		weights = phase_weights[phase]

		# Calculate heuristic score
		score = (
				weights['number_occupied_spaces'] * number_occupied_spaces +
				weights['opponents_taken_pieces'] * opponents_taken_pieces +
				weights['sum_distances'] * sum_distances +
				weights['sum_distances_opponent'] * sum_distances_opponent +
				weights['number_of_singles'] * number_of_singles +
				weights['sum_single_distance_away_from_home'] * sum_single_distance_away_from_home +
				weights['pieces_on_board'] * pieces_on_board +
				weights['sum_distances_to_endzone'] * sum_distances_to_endzone +
				weights['finished_pieces'] * finished_pieces
		)

		return score
	
	def add_child(self, child):
		self.children.append(child)
	
	@staticmethod
	def recursive_get_all_possible_moves(board, colour, dice_rolls, current_secuence = [], time_limit=None):
		if not dice_rolls:
			return [current_secuence]
		current_dice = dice_rolls[0]
		all_moves = []
		for piece in board.get_pieces(colour):
			if time_limit is not None and time.time() >= time_limit:
				return 
			if board.is_move_possible(piece, current_dice):
				board_copy = board.create_copy()
				new_piece = board_copy.get_piece_at(piece.location)
				board_copy.move_piece(new_piece, current_dice)
				all_moves.extend(MinimaxNode.recursive_get_all_possible_moves(board_copy, colour, dice_rolls[1:], current_secuence + [(piece.location, current_dice)]))
		return all_moves
	
	def expend(self, time_limit=None):
		if time_limit is not None and time.time() >= time_limit:
			return

		if self.depth == self.max_depth or self.board.has_game_ended():
			return
		
		if self.mode == "max" or self.mode == "min":
			all_board_states = set()
			all_moves = MinimaxNode.recursive_get_all_possible_moves(self.board, self.colour, self.dice_rolls, time_limit=time_limit)
			if not all_moves:
				return
			for moves in all_moves:
				if time_limit is not None and time.time() >= time_limit:
					return
				board_copy = self.board.create_copy()
				i = 0
				for move in moves:
					piece = move[0]
					dice_roll = move[1]
					new_piece = board_copy.get_piece_at(piece)
					board_copy.move_piece(new_piece, dice_roll)
				if board_copy not in all_board_states:
					all_board_states.add(board_copy)
					child = MinimaxNode(board_copy, self.colour.other(), [], self.depth + 1, self.max_depth, moves, self, "avg")
					self.add_child(child)
		else:
			all_possible_dice_rolls = [[x, y] for x in range(1, 7) for y in range(1, 7) if x != y]
			all_possible_dice_rolls = all_possible_dice_rolls + [[x, x, x, x] for x in range(1, 7)]
			for dice_roll in all_possible_dice_rolls:
				parent_mode = self.parent.mode
				child_mode = "max" if parent_mode == "min" else "min"
				child = MinimaxNode(self.board, self.colour, dice_roll, self.depth, self.max_depth, None, self, child_mode)
				self.add_child(child)
	
	def evaluate(self, alpha=float('-inf'), beta=float('inf'), time_limit=None):
		if time_limit is not None and time.time() >= time_limit:
			return self.heuristic_function(self.assess_board(self.colour, self.board))

		if self.depth == self.max_depth or self.board.has_game_ended():
			state = self.assess_board(self.colour, self.board)
			score = self.heuristic_function(state)
			return score
		
		self.expend(time_limit=time_limit)	

		if not self.children:
			return self.heuristic_function(self.assess_board(self.colour, self.board))
		
		if self.mode == "max":
			best_score = float('-inf')
			for child in self.children:
				score = child.evaluate(alpha, beta, time_limit)
				if score > best_score:
					best_score = score
				alpha = max(alpha, best_score)
				if beta <= alpha:
					break
			self.best_score = best_score
			return best_score

		elif self.mode == "min":
			best_score = float('inf')
			for child in self.children:
				score = child.evaluate(alpha, beta, time_limit)
				if score < best_score:
					best_score = score
				beta = min(beta, best_score)
				if beta <= alpha:
					break
			self.best_score = best_score
			return best_score

		else:
			best_score = 0
			for child in self.children:
				score = child.evaluate(alpha, beta, time_limit)
				best_score += score
			self.best_score = best_score / len(self.children) if self.children else self.heuristic_function(self.assess_board(self.colour, self.board))
			return self.best_score

class MinimaxStrategy(Strategy):
	

	@staticmethod
	def get_difficulty():
		return "Hard"

	def __init__(self, depth=2):
		super().__init__()
		self.max_depth = depth
	
	def move(self, board, colour, dice_roll, make_move, opponents_activity):
		time_limit = time.time() + board.getTheTimeLim() - 4
		root = MinimaxNode(board, colour, dice_roll, 0, self.max_depth)
		root.expend(time_limit=time_limit)
		best_score = float('-inf')
		best_moves = None
		for child in root.children:
			score = child.evaluate(time_limit=time_limit)
			if score > best_score:
				best_score = score
				best_moves = child.move
		if best_moves is None:
			return
		for move in best_moves:
			piece = move[0]
			dice_roll = move[1]
			make_move(piece, dice_roll)
		return