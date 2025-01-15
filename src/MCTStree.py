import math
import random
from itertools import permutations
from src.strategies import Strategy
import time

class Huristic(Strategy):

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

	def heuristic_function(self, state_stats, player, current_player, node):
		"""
		Evaluate the board state_stats and return a heuristic score.

		Args:
			state_stats (dict): A dictionary containing game state_stats metrics.

		Returns:
			float: A heuristic score for the current board state.
		"""
		if node.state.has_game_ended():
			if node.state.who_won() == player:
				return float('inf')
			else:
				return float('-inf')
		elif not node.has_valid_moves:
			return -100
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
		average_distance_to_home = sum_distances_to_endzone / pieces_out_of_home if pieces_out_of_home > 0 else 0

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
				'number_occupied_spaces': 1.0,
				'opponents_taken_pieces': 1.5,
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
				'number_of_singles': -3.0,
				'pieces_on_board': 1.0,
				'sum_distances_to_endzone': -0.7,
				'sum_single_distance_away_from_home': -0.3,  # Heavily penalize singles far from home
				'finished_pieces': 4.0,
			},
			'late': {
				'number_occupied_spaces': 0.5,
				'opponents_taken_pieces': 2.5,
				'sum_distances': -0.3,
				'sum_distances_opponent': 0.2,
				'number_of_singles': -1.0,
				'pieces_on_board': 1.5,
				'sum_distances_to_endzone': -1.0,
				'sum_single_distance_away_from_home': -0.1,  # Heavily penalize singles far from home
				'finished_pieces': 7.0,
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
	
class MCTSNode:
	def __init__(self, state,player, current_player, parent=None, action = None, possible_dice_rolls=None):
		"""
		Initialize a node in the MCTS tree.

		Args:
			state: The game state represented by this node.
			player: The root player color.
			current_player: The current player color.
			dice_rolls (list): The dice rolls available for this state.
			parent (MCTSNode): Reference to the parent node.
		"""
		self.state = state
		self.player = player
		self.current_player = current_player
		if possible_dice_rolls is None:
			self.possible_dice_rolls = [[i, j] for i in range(1, 7) for j in range(1, 7) if i != j]
			self.possible_dice_rolls += [[i, i, i, i] for i in range(1, 7)]
		else:
			self.possible_dice_rolls = possible_dice_rolls
		self.parent = parent
		self.action = action
		self.children = []
		self.move_generator = None
		self.visit_count = 0
		self.total_reward = 0.0
		self.has_valid_moves = True



	@staticmethod
	def apply_move(board, move):
		piece = move[0]
		die_roll = move[1]
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
		board = board or self.state
		if dice_rolls is None:
			dice_rolls = random.choice(self.possible_dice_rolls)
			self.possible_dice_rolls.remove(dice_rolls)
		else:
			dice_rolls = dice_rolls
		move_sequence = move_sequence or []

		if not dice_rolls or board.has_game_ended():  # Base case: no more dice to use
			yield move_sequence
			return
		
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



	def is_fully_expanded(self):
		"""
		Check if the node is fully expanded.

		Returns:
			bool: True if all actions have been tried, False otherwise.
		"""
		return len(self.possible_dice_rolls) == 0 and self.move_generator is None

	def best_child(self, exploration_constant=1.4):
		"""
		Select the best child node based on the UCB1 formula.

		Args:
			exploration_constant (float): The exploration parameter (C).

		Returns:
			MCTSNode: The best child node.
		"""
		if not self.children:
			self.has_valid_moves = False
			return self
		return max(
			self.children,
			key=lambda child: (
				(-1 if self.current_player != self.player else 1) *
				child.total_reward / child.visit_count +
				exploration_constant * math.sqrt(
					math.log(self.visit_count) / child.visit_count
				)
				if child.visit_count > 0 else (float('inf') if self.current_player == self.player else float('-inf'))
			)
		)

	def add_child(self, action):
		"""
		Add a child node for the given action and state.

		Args:
			action: The action leading to the child node.
			new_state: The resulting state from applying the action.
			dice_rolls (list): The dice rolls for the new state.

		Returns:
			MCTSNode: The newly created child node.
		"""
		board = self.state
		for move in action:
			board = self.apply_move(board, move)
		child_node = MCTSNode(state=board, parent=self, player=self.player, current_player=self.current_player.other(), action=action)
		self.children.append(child_node)
		return child_node

	def update(self, reward):
		"""
		Update the node's statistics with the given reward.

		Args:
			reward (float): The reward to add to the total reward.
		"""
		self.visit_count += 1
		self.total_reward += reward

	def simulate(self):
		"""
		evaluate the current state using a heuristic function.
		"""
		huristic = Huristic()
		state_stats = huristic.assess_board(self.player, self.state)
		return huristic.heuristic_function(state_stats, self.player, self.current_player, self)


class MCTSTreeStrategy(Strategy):
	@staticmethod
	def get_difficulty():
		return "Hard"		


	def tree_policy(self, root, limit_buffered, start_time):
		"""
		Traverse the tree to select a node for expansion using the Tree Policy.

		Returns:
			MCTSNode: The selected node.
		"""
		current_node = root
		while not current_node.state.has_game_ended() and time.time() - start_time < limit_buffered:
			if not current_node.is_fully_expanded():
				node = self.expand(current_node)
				if node is not None:
					return node
				else:
					continue
			else:
				current_node = current_node.best_child()
		return current_node
	
	def Backup(self, node, reward):
		"""
		Update the statistics of the nodes in the path from the given node to the root.

		Args:
			node (MCTSNode): The node to start the backup from.
			reward (float): The reward to backpropagate.
		"""
		while node is not None:
			node.update(reward)
			node = node.parent

	def expand(self, node):
		"""
		Expand the given node by adding a child for an untried action.

		Args:
			node (MCTSNode): The node to expand.

		Returns:
			MCTSNode: The newly expanded child node.
		"""
		if node.move_generator is None:
			dice_roll = random.choice(node.possible_dice_rolls)
			node.possible_dice_rolls.remove(dice_roll)
			move_generator = node.generate_possible_moves(dice_rolls=dice_roll)
			node.move_generator = move_generator
		
		try:
			action = next(node.move_generator)
		except StopIteration:
			node.move_generator = None
			return None
		return node.add_child(action)
	
	def move(self, board, player, dice_rolls, make_move, opponents_activity):

		# Start tracking time to ensure we do not exceed the time limit
		time_limit = board.getTheTimeLim()
		start_time = time.time()
		limit_buffered = time_limit - 0.2  # Small buffer to ensure we stay within the time limit
		updating_iter_time = 0

		# Create root node v0 woth state s0
		root = MCTSNode(board, player, player, possible_dice_rolls=[dice_rolls])

		while time.time() - start_time < limit_buffered:
			iter_start_time = time.time()
			# Selection
			v1 = self.tree_policy(root, limit_buffered, start_time)

			if v1.state.has_game_ended():
				break

			# Simulation
			reward = v1.simulate()

			# Backpropagation
			self.Backup(v1, reward)
			iter_end_time = time.time()
			updating_iter_time = 0.5 * (iter_end_time - iter_start_time) + 0.5 * updating_iter_time if updating_iter_time != 0 else iter_end_time - iter_start_time
			limit_buffered = time_limit - updating_iter_time - 0.2
		
		# Select the best child of the root node
		best_child = root.best_child(exploration_constant=0.0)
		best_action = best_child.action
		for move in best_action:
			print("************")
			print("moving: ", move[0].location, move[1])
			print("************")
			make_move(move[0].location, move[1])
