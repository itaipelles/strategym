import copy
import numpy as np
from enum import Enum
from collections import namedtuple

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
    def get_owners(self):
        return np.array([territory.owner.value for territory in self.territories])
    def get_player_infantry(self, player:Players):
        result = np.array([territory.units[player][Units.INFANTRY.value] for territory in self.territories])
        return result
    def board_scores(self):
        scores = np.zeros(shape=(len(Players),))
        for player in Players:
            territories_income = 0
            units_value = 0
            for territory in self.territories:
                units_value += territory.units[player][Units.INFANTRY.value]*UNITS_STATS[Units.INFANTRY].cost
                if territory.owner == player:
                    territories_income += territory.income
            
            scores[player.value] = units_value + territories_income

        return scores