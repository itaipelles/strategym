import random
from axis_and_allies_game.board import Units, UNITS_STATS, Players, Territory

def get_dmg(attack_value :int):
    dmg = attack_value // 6
    left_over_attack = attack_value % 6
    dmg += (random.randint(1, 6) <= left_over_attack)
    return dmg

def infantry_battle(attacker_inf : int, defender_inf : int):
    while attacker_inf>0 and defender_inf>0:
        attack_dmg = get_dmg(attacker_inf*UNITS_STATS[Units.INFANTRY].attack)
        defence_dmg = get_dmg(defender_inf*UNITS_STATS[Units.INFANTRY].defense)
        attacker_inf = max(0, attacker_inf-defence_dmg)
        defender_inf = max(0, defender_inf-attack_dmg)
    return attacker_inf, defender_inf

def resolve_fight(territory:Territory, attacker:Players, alliances:dict):
    
    defenders:list[Players] = []
    defenders += []
    for player in Players:
        if alliances[attacker] != alliances[player]:
            defenders += [player]
    if len(defenders) == 0:
        territory.owner = attacker
        return
    atk_inf = territory.units[attacker][Units.INFANTRY.value]
    dfnd_inf = territory.units[defenders[0]][Units.INFANTRY.value]
    new_attack_inf, new_defend_inf = infantry_battle(atk_inf, dfnd_inf)
    territory.units[attacker][Units.INFANTRY.value] = new_attack_inf
    territory.units[defenders[0]][Units.INFANTRY.value] = new_defend_inf
    if new_attack_inf > 0:
        territory.owner = attacker
    return


if __name__ == '__main__':
    attacker_inf = 10
    defender_inf = 5
    t = Territory(Players.GERMANY, 0, {Players.GERMANY: [14,0], Players.RUSSIA: [12,0]})
    resolve_fight(t, Players.RUSSIA, {})
    print(t.units[Players.GERMANY])
    print(t.units[Players.RUSSIA])