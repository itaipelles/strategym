import copy
import numpy as np
from board import *
from axisAndAllies_game.gameRenderer import GameRenderer
from battleCalculator import BattleCalculator

class Game():
    board:Board
    starting_board_state:Board
    round_playing_order:list[Players]
    current_player_turn:Players
    round_counter:int
    win_condition:int
    alliances:dict[Players,str]
    current_move:list[int]
    illegal_moves_count:int

    def __init__(self, board:Board, round_playing_order:list[Players], win_condition:int, alliances:dict):
        self.starting_board_state = copy.deepcopy(board)
        self.round_playing_order = copy.deepcopy(round_playing_order)
        self.win_condition = copy.deepcopy(win_condition)
        self.alliances = copy.deepcopy(alliances)
        self.reset()

    def reset(self):
        self.board = copy.deepcopy(self.starting_board_state)
        self.round_counter = 0
        self.current_player_turn = self.round_playing_order[0]
        self.current_move = [0]*self.board.num_of_adjacencies()
        self.illegal_moves_count = 0

    def play_turn(self, action:np.ndarray[float]) -> tuple[bool, int]:
        self.illegal_moves_count = 0
        if self.current_player_turn == self.round_playing_order[0]:
            self.round_counter += 1
        
        # translate continuous action into discrete
        self.current_move = []
        for percentage, (from_territory, to_territory) in zip(action, self.board.adjacencies):
            infantry_to_move = np.round(percentage * self.board.territories[from_territory].units[self.current_player_turn][Units.INFANTRY.value])
            self.current_move.append(infantry_to_move)
        self.current_move = np.array(self.current_move, dtype=int)

        # action legality
        for idx, territory in enumerate(self.board.territories):
            infantry_amount = territory.units[self.current_player_turn][Units.INFANTRY.value]
            adjacency_indices = self.board.adjacencies[:, 0] == idx
            sum_of_infantry_leaving_territory = np.sum(self.current_move[adjacency_indices])
            if sum_of_infantry_leaving_territory > infantry_amount:
                self.current_move[adjacency_indices] = (self.current_move[adjacency_indices] * infantry_amount) // sum_of_infantry_leaving_territory
                self.illegal_moves_count += sum_of_infantry_leaving_territory - infantry_amount

        # move infantries
        outgoing_infantries = np.bincount(self.board.adjacencies[:, 0], self.current_move, minlength=self.board.num_of_territories()).astype(int)
        incoming_infantries = np.bincount(self.board.adjacencies[:, 1], self.current_move, minlength=self.board.num_of_territories()).astype(int)
        contested_territories:list[Territory] = []
        for idx, territory in enumerate(self.board.territories):
            territory.units[self.current_player_turn][Units.INFANTRY.value] += incoming_infantries[idx] - outgoing_infantries[idx]
            if not self.are_allies(self.current_player_turn, territory.owner):
                contested_territories.append(territory)

        # resolve fights
        for territory in set(contested_territories):
            BattleCalculator.resolve_fight(territory=territory, attacker=self.current_player_turn, alliances=self.alliances)

        # reinforcements
        income = np.sum([territory.income for territory in self.board.territories if territory.owner == self.current_player_turn])
        new_infantry_amount = np.floor(income/UNITS_STATS[Units.INFANTRY].cost)
        spawn_points = self.board.capitals.intersection([i for i, territory in enumerate(self.board.territories) if territory.owner == self.current_player_turn])
        if bool(spawn_points):
            self.board.territories[next(iter(spawn_points))].units[self.current_player_turn][Units.INFANTRY.value] += new_infantry_amount

        # check for winning
        alliance_capital_control = [territory for idx, territory in enumerate(self.board.territories) if (idx in self.board.capitals and self.are_allies(self.current_player_turn, territory.owner))]

        # passing the turn to the next player
        idx = self.round_playing_order.index(self.current_player_turn) + 1
        self.current_player_turn = self.round_playing_order[idx] if idx<len(self.round_playing_order) else self.round_playing_order[0] 

        if len(alliance_capital_control) >= self.win_condition:
            return True, self.illegal_moves_count
        return False, self.illegal_moves_count
    
    def are_allies(self,player1:Players, player2:Players):
        return self.alliances[player1] == self.alliances[player2]

def set_game():
    # set up board
    owners = [Players.RUSSIA, Players.RUSSIA, Players.RUSSIA, Players.GERMANY, Players.GERMANY, Players.GERMANY]
    incomes = [6, 3, 3, 3, 3, 6]
    units = [{Players.RUSSIA:[10,0]},{Players.RUSSIA:[2,0]},{Players.RUSSIA:[2,0]},
            {Players.GERMANY:[2,0]},{Players.GERMANY:[2,0]},{Players.GERMANY:[11,0]}]
    territories = [Territory(owner, income, unit) for owner,income,unit in zip(owners, incomes, units)]
    adjacencies = [(0,1), (0,2), (1,3), (2,4), (3,5), (4,5)]
    capitals = [0, 5]
    board = Board(territories=territories, adjacencies=adjacencies, capitals=capitals)
    # set up game
    round_playing_order = [Players.RUSSIA, Players.GERMANY]
    alliances = {Players.RUSSIA:'Allies', Players.GERMANY:'Axis'}
    game = Game(board=board, round_playing_order=round_playing_order, win_condition=2, alliances=alliances)
    return game

def set_game_v2():
    # set up board
    owners = [Players.RUSSIA]*8 + [Players.GERMANY]*8
    incomes = [6]*2 + [3]*12 + [6]*2
    units = [{owner: [2, 0]} if incomes[i] == 3 else {owner: [10, 0]} for i, owner in enumerate(owners)]
    territories = [Territory(owner, income, unit) for owner,income,unit in zip(owners, incomes, units)]

    adjacencies = [(0,2), (2,1), (0,3), (0,4), (1,5), (1,6), (3,8), (4,7), (5,7), (6,9), (7,10), (10,11), (10,12), (8,14), (11,14), (12,15), (9,15), (14,13), (15,13)]
    capitals = [0, 1, 14,15]
    board = Board(territories=territories, adjacencies=adjacencies, capitals=capitals)
    # set up game
    round_playing_order = [Players.RUSSIA, Players.GERMANY]
    alliances = {Players.RUSSIA:'Allies', Players.GERMANY:'Axis'}
    game = Game(board=board, round_playing_order=round_playing_order, win_condition=3, alliances=alliances)
    return game

# run game
if __name__ == "__main__":
    from gymnasium import spaces
    game = set_game_v2()
    renderer = GameRenderer(game.board)
    renderer.render(game.board,game.current_move,game.illegal_moves_count>0)
    action_space = spaces.Box(low=0,high=1,shape=(len(game.board.adjacencies),))
    for i in range(100):
        action = action_space.sample()
        victory,_ = game.play_turn(action)
        renderer.render(game.board,game.current_move,game.illegal_moves_count>0)
        if victory:
            game.reset()
            renderer.render(game.board,game.current_move,game.illegal_moves_count>0)