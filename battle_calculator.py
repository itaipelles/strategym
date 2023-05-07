import numpy as np
import random

INFANTRY_ATTACK_VALUE = 2
INFANTRY_DEFENCE_VALUE = 2


def get_dmg(attack_value :int):
    dmg = attack_value // 6
    left_over_attack = attack_value % 6
    dmg += (random.randint(1, 6) <= left_over_attack)
    return dmg

def infantry_battle(attacker_inf : int, defender_inf : int):
    while attacker_inf>0 and defender_inf>0:
        attack_dmg = get_dmg(attacker_inf*INFANTRY_ATTACK_VALUE)
        defence_dmg = get_dmg(defender_inf*INFANTRY_DEFENCE_VALUE)
        attacker_inf = max(0, attacker_inf-defence_dmg)
        defender_inf = max(0, defender_inf-attack_dmg)
    return attacker_inf, defender_inf

