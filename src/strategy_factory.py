from src.compare_all_moves_strategy import CompareAllMovesSimple, HuristicCompareAllMoves
from src.strategies import MoveFurthestBackStrategy, HumanStrategy, MoveRandomPiece
from src.MCTStree import MCTSTreeStrategy
from src.Minimax import MinimaxStrategy


class StrategyFactory:
    @staticmethod
    def create_by_name(strategy_name):
        for strategy in StrategyFactory.get_all():
            if strategy.__name__ == strategy_name:
                return strategy()

        raise Exception("Cannot find strategy %s" % strategy_name)

    @staticmethod
    def get_all():
        strategies = [
            MoveRandomPiece,
            MoveFurthestBackStrategy,
            CompareAllMovesSimple,
            HumanStrategy,
            HuristicCompareAllMoves,
            MinimaxStrategy,
			MCTSTreeStrategy
        ]
        return strategies
