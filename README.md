# README
315350868

## Overview
This project is a platform for simulating a tournament of backgammon strategies. It allows users to implement and test various AI strategies in a competitive environment.

## Project Structure
The codebase consists of the following key components:

### Core Modules
- **`tournament.py`**:
  - Manages the tournament flow, including rounds, matchups, and determining the winner.
  - Provides logging for tournament progress and results.

- **`game.py`**:
  - Handles the logic of a single game, including player turns, dice rolls, and move validation.
  - Uses the `Board` class to manage the game state and checks for the end of the game.

- **`board.py`**:
  - Implements the board representation and handles piece movements.
  - Provides functions for validating moves, tracking piece locations, and checking game-ending conditions.

### Strategy Modules
- **`strategies.py`**:
  - Defines base and common strategies for players.
  - Includes examples like `MoveFurthestBackStrategy` and `MoveRandomPiece`.

- **`compare_all_moves_strategy.py`**:
  - Implements advanced strategies that evaluate the board state and compute optimal moves.
  - Includes your custom strategies:
    - **`HuristicCompareAllMoves`**: Evaluates board states using a heuristic function.
    - **`MinimaxPlayer`**: Implements a Minimax algorithm for decision-making, considering future moves up to a specified depth.

- **`strategy_factory.py`**:
  - Provides a factory for creating strategies by name.
  - Includes registration for the newly added `HuristicCompareAllMoves` and `MinimaxPlayer` strategies.

## Contributions
315350868
### Additions by [Ben Rotlevy]
1. **`compare_all_moves_strategy.py`**:
   - Added the `HuristicCompareAllMoves` strategy, utilizing a heuristic evaluation function for smarter decision-making.
   - Added the `MinimaxPlayer` strategy, implementing a Minimax algorithm for planning moves up to a defined depth.

2. **`strategy_factory.py`**:
   - Registered the new strategies (`HuristicCompareAllMoves` and `MinimaxPlayer`) in the factory to make them available for use in tournaments.

### Original Author
- The rest of the platform, including the core modules and initial strategy implementations, was developed by **Shay Uzan**.

## How to Use
1. **Run the Tournament**
   - Execute `tournament.py` to start a tournament.
   - Follow the prompts to input player names and assign strategies.

2. **Add Custom Strategies**
   - Implement a new strategy by extending the `Strategy` base class or using existing advanced strategies as templates.
   - Add the new strategy to `strategy_factory.py` to include it in the available options.

3. **Experiment**
   - Modify the game parameters, such as time limits and dice roll mechanics, to explore different scenarios.

## Requirements
- Python 3.8 or later
- Additional dependencies listed in `requirements.txt` (if applicable)

## Future Improvements
- Enhance the user interface for better visualization of games.
- Add support for more complex board configurations.
- Optimize the Minimax implementation for deeper look-ahead.

## License
This project is distributed under [LICENSE_NAME].

