import copy, random
import numpy as np
from enum import Enum
from collections import namedtuple
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

class Players(Enum):
    RUSSIA = 0
    GERMANY = 1

class Units(Enum):
    INFANTRY  = 0
    ARTILLERY = 1

Stats = namedtuple("stats", "attack defense movement cost")
UNITS_STATS = {Units.INFANTRY:Stats(2,2,1,3),
               Units.ARTILLERY:Stats(2,2,1,4)}

class Territory():
    owner:Players
    income:int
    units:dict
    def __init__(self, owner:Players, income:int = 0, units:dict = {}):
        self.owner = owner
        self.income = income
        self.units = {}
        for player in Players:
            if player in units:
                self.units[player] = np.array(units[player])
            else:
                self.units[player] = np.zeros(len(Units.__members__), dtype=int)

class Board():
    territories_id:list[int]
    territories:list[Territory]
    adjacencies:np.ndarray[tuple[int,int]]
    capitals:set[int]
    def __init__(self, territories:list[Territory], adjacencies:list[tuple[int,int]], capitals:set[int]):

        self.territories = territories
        self.adjacencies = np.array(adjacencies + [(j,i) for (i,j) in adjacencies if (j,i) not in adjacencies])
        self.capitals = copy.deepcopy(set(capitals))
        self.territories_id = range(len(self.territories))
    def num_of_territories(self):
        return len(self.territories)
    def num_of_adjacencies(self):
        return len(self.adjacencies)

class BattleCalcultor():
    def get_dmg(attack_value :int):
        dmg = attack_value // 6
        left_over_attack = attack_value % 6
        dmg += (random.randint(1, 6) <= left_over_attack)
        return dmg

    def infantry_battle(attacker_inf : int, defender_inf : int):
        while attacker_inf>0 and defender_inf>0:
            attack_dmg = BattleCalcultor.get_dmg(attacker_inf*UNITS_STATS[Units.INFANTRY].attack)
            defence_dmg = BattleCalcultor.get_dmg(defender_inf*UNITS_STATS[Units.INFANTRY].defense)
            attacker_inf = max(0, attacker_inf-defence_dmg)
            defender_inf = max(0, defender_inf-attack_dmg)
        return attacker_inf, defender_inf
    
    def resolve_fight(territory:Territory, attacker:Players, alliances:dict):
        defenders:list[Players] = []
        for player in Players:
            if alliances[attacker] != alliances[player]:
                defenders += [player]
        if len(defenders) == 0:
            territory.owner = attacker
            return
        atk_inf = territory.units[attacker][Units.INFANTRY.value]
        dfnd_inf = territory.units[defenders[0]][Units.INFANTRY.value]
        new_attack_inf, new_defend_inf = BattleCalcultor.infantry_battle(atk_inf, dfnd_inf)
        territory.units[attacker][Units.INFANTRY.value] = new_attack_inf
        territory.units[defenders[0]][Units.INFANTRY.value] = new_defend_inf
        if new_attack_inf > 0:
            territory.owner = attacker
        return
    
class Game():
    board:Board
    starting_board_state:Board
    round_playing_order:list[Players]
    current_player_turn:Players
    round_counter:int
    win_condition:int
    alliances:dict[Players,str]
    current_move:list[int]
    G:nx.Graph
    illegal_moves_count:int

    def __init__(self, board:Board, round_playing_order:list[Players], win_condition:int, alliances:dict):
        self.starting_board_state = copy.deepcopy(board)
        self.round_playing_order = copy.deepcopy(round_playing_order)
        self.win_condition = copy.deepcopy(win_condition)
        self.alliances = copy.deepcopy(alliances)
        self.reset()

        self.G = nx.Graph()
        self.G.add_nodes_from(self.board.territories_id)
        self.G.add_edges_from(board.adjacencies)
        self.G_node_pos = nx.spring_layout(self.G)
        

    def reset(self):
        self.board = copy.deepcopy(self.starting_board_state)
        self.round_counter = 0
        self.current_player_turn = self.round_playing_order[0]
        self.current_move = [0]*self.board.num_of_adjacencies()
        self.illegal_moves_count = 0

    def step(self, action:np.ndarray[float]) -> tuple[bool, int]:
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
            BattleCalcultor.resolve_fight(territory=territory, attacker=self.current_player_turn, alliances=self.alliances)

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
    
    def get_owners(self):
        return np.array([territory.owner.value for territory in self.board.territories])
    
    def get_player_infantry(self, player:Players):
        result = np.array([territory.units[player][Units.INFANTRY.value] for territory in self.board.territories])
        return result
    
    def are_allies(self,player1:Players, player2:Players):
        return self.alliances[player1] == self.alliances[player2]
    
    def boardScores(self):
        scores = np.zeros(shape = (len(Players),))
        for player in Players:
            scores[player.value] = np.sum([territory.income for territory in self.board.territories if territory.owner == player])
            for territory in self.board.territories:
                scores[player.value] += territory.units[player][Units.INFANTRY.value]*UNITS_STATS[Units.INFANTRY].cost
        
        return scores
    
    def render(self):
        # the commented lines might be needed when we scale to more types of units. for now i print the infantries directly
        # nx.set_node_attributes(G, dict(zip(territories,obs['player1_infantry'])), name = 'p1_infantry')
        # nx.set_node_attributes(G, dict(zip(territories,obs['player2_infantry'])), name = 'p2_infantry')
        
        fig = plt.figure()
        plt.axis('off')
        plt.tight_layout()

        G_p1_pos = self.G_node_pos.copy()
        G_p2_pos = self.G_node_pos.copy()
        for i, key in (self.G_node_pos.items()):
            G_p1_pos[i] = self.G_node_pos[i] + [0, 0.15]
            G_p2_pos[i] = self.G_node_pos[i] - [0, 0.15]
        
        color_map = []
        for territory in self.board.territories:
            if territory.owner==Players.RUSSIA:
                color_map.append('red')
            elif territory.owner==Players.GERMANY:
                color_map.append('black')

        nx.draw_networkx(self.G, self.G_node_pos, with_labels = True, 
                node_color = color_map, node_size=500,
                font_color = "white", font_size = 15)
        nx.draw_networkx_labels(self.G,G_p1_pos, labels = dict(zip(self.board.territories_id,self.get_player_infantry(Players.RUSSIA))), font_color = "red")
        nx.draw_networkx_labels(self.G,G_p2_pos, labels = dict(zip(self.board.territories_id,self.get_player_infantry(Players.GERMANY))), font_color = "black")

        edge_labels = {tuple(sorted(x)) : 0 for x in self.board.adjacencies}
        for infantry_to_move, (from_territory, to_territory) in zip(self.current_move, self.board.adjacencies):
            if(to_territory < from_territory):
                edge_labels[(to_territory,from_territory)] -= infantry_to_move
            else:
                edge_labels[(from_territory,to_territory)] += infantry_to_move

        nx.draw_networkx_edge_labels(
            self.G, self.G_node_pos,
            edge_labels=edge_labels,
            font_color='green')

        fig.set_facecolor("white")
        if(self.illegal_moves_count > 0):
            fig.set_facecolor("gray")

        fig.canvas.draw()
        plt.show()
        return

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

# run game
if __name__ == "__main__":
    from gymnasium import spaces
    game = set_game()
    game.render()
    action_space = spaces.Box(low=0,high=1,shape=(len(game.board.adjacencies),))
    for i in range(100):
        action = action_space.sample()
        victory,_ = game.step(action)
        game.render()
        if victory:
            game.reset()
            game.render()