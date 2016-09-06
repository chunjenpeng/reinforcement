#class Object:

#class Relation:
def isNear(pos1, pos2, near):
    from util import manhattanDistance
    if manhattanDistance(pos1, pos2) <= near:
        return True
    return False

def atEastOf(pos1, pos2):
    if pos1[0] > pos2[0]:
        return True
    return False

def atNorthOf(pos1, pos2):
    if pos1[1] > pos2[1]:
        return True
    return False

class Feature:
    def __init__(self, name, closure):
        self.func = closure
        self.name = name
    def __str__(self):
        return self.name
    def satisfy(self, gameState):
        return self.func(gameState)

def Ghost_isNear_Pacman(gameState):
    return isNear( gameState.getGhostPosition(1), gameState.getPacmanPosition(), near = 1 )

def Ghost_atEastOf_Pacman(gameState):
    return atEastOf(gameState.getGhostPosition(1), gameState.getPacmanPosition())

def Ghost_atNorthOf_Pacman(gameState):
    return atNorthOf(gameState.getGhostPosition(1), gameState.getPacmanPosition())

def generateFeatures():
    features = [] 
    features.append( Feature('Ghost_isNear_Pacman', Ghost_isNear_Pacman))
    features.append( Feature('Ghost_atEastOf_Pacman', Ghost_atEastOf_Pacman))
    features.append( Feature('Ghost_atNorthOf_Pacman', Ghost_atNorthOf_Pacman))
    return features

def satisfyFeatures(chromosome, features, gameState):
    for i in xrange(len(features)):
        if chromosome[i] == '*':continue
        if chromosome[i] == '0' and features[i].satisfy(gameState):
            return False 
        if chromosome[i] == '1' and not features[i].satisfy(gameState):
            return False 
    return True
