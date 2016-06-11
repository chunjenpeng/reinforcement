from random import randint

class Rule:
    def __init__(self, closure):
        self.func = closure
    def check(self):
        self.func()

def


maxLength = 10
mazeLength = randint(3, maxLength)
#mazeWidth = 3
pacmanLocation = 2
ghostLocation = 3
foodLocations = []



def rule_ghost(direction, distance):
    if direction == 'right':
        


def rule2states(rule):
    


def generateState(rules):
    for rule in rules:
        state = rule2states(rule)
        
