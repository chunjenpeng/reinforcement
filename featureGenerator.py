from featureExtractors import closestFoodPosition
#class Object:

#class Relation:
def isNear(pos1, pos2, near):
    from util import manhattanDistance
    if manhattanDistance(pos1, pos2) <= near:
        return True
    return False

def atEastOf(pos1, pos2):
    return True if pos1[0] > pos2[0] else False

def atWestOf(pos1, pos2):
    return True if pos1[0] < pos2[0] else False

def atNorthOf(pos1, pos2):
    return True if pos1[1] > pos2[1] else False

def atSouthOf(pos1, pos2):
    return True if pos1[1] < pos2[1] else False

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

def Ghost_atWestOf_Pacman(gameState):
    return atWestOf(gameState.getGhostPosition(1), gameState.getPacmanPosition())

def Ghost_atNorthOf_Pacman(gameState):
    return atNorthOf(gameState.getGhostPosition(1), gameState.getPacmanPosition())

def Ghost_atSouthOf_Pacman(gameState):
    return atSouthOf(gameState.getGhostPosition(1), gameState.getPacmanPosition())


def ClosestFood_isNear_Pacman(gameState):
    from featureExtractors import closestFood
    dist = closestFood( gameState.getPacmanPosition(), gameState.getFood(), gameState.getWalls())
    if dist is not None:
        return(dist == 1)
    return False

def ClosestFood_atEastOf_Pacman(gameState):
    closestFood = closestFoodPosition( gameState.getPacmanPosition(), gameState.getFood(), gameState.getWalls())
    return atEastOf(closestFood, gameState.getPacmanPosition())

def ClosestFood_atWestOf_Pacman(gameState):
    closestFood = closestFoodPosition( gameState.getPacmanPosition(), gameState.getFood(), gameState.getWalls())
    return atWestOf(closestFood, gameState.getPacmanPosition())

def ClosestFood_atNorthOf_Pacman(gameState):
    closestFood = closestFoodPosition( gameState.getPacmanPosition(), gameState.getFood(), gameState.getWalls())
    return atNorthOf(closestFood, gameState.getPacmanPosition())

def ClosestFood_atSouthOf_Pacman(gameState):
    closestFood = closestFoodPosition( gameState.getPacmanPosition(), gameState.getFood(), gameState.getWalls())
    return atSouthOf(closestFood, gameState.getPacmanPosition())

def generateFeatures():
    features = []
    #'''
    features.append( Feature('Ghost_isNear_Pacman', Ghost_isNear_Pacman))
    features.append( Feature('Ghost_atEastOf_Pacman', Ghost_atEastOf_Pacman))
    features.append( Feature('Ghost_atWestOf_Pacman', Ghost_atWestOf_Pacman))
    features.append( Feature('Ghost_atNorthOf_Pacman', Ghost_atNorthOf_Pacman))
    features.append( Feature('Ghost_atSouthOf_Pacman', Ghost_atSouthOf_Pacman))
    #''' 
    features.append( Feature('ClosestFood_isNear_Pacman', ClosestFood_isNear_Pacman))
    features.append( Feature('ClosestFood_atEastOf_Pacman', ClosestFood_atEastOf_Pacman))
    features.append( Feature('ClosestFood_atWestOf_Pacman', ClosestFood_atWestOf_Pacman))
    features.append( Feature('ClosestFood_atNorthOf_Pacman', ClosestFood_atNorthOf_Pacman))
    features.append( Feature('ClosestFood_atSouthOf_Pacman', ClosestFood_atSouthOf_Pacman))
    #''' 
    return features


def satisfyFeatures(chromosome, features, gameState):
    for i in xrange(len(features)):
        if chromosome[i] == '*':continue
        if chromosome[i] == '0' and features[i].satisfy(gameState):
            return False 
        if chromosome[i] == '1' and not features[i].satisfy(gameState):
            return False 
    return True
