#author Daniel & Ryan
import featureExtractors
import util, sys
from random import randint
from random import seed

class gameData:
    def __init__(self, args):
        self.mazeHeight = args['mazeHeight']
        self.mazeLength = args['mazeLength']
        if args['posPacman'] is None:
            self.posPacman = randint(1, self.mazeLength)
        else:
            self.posPacman = args['posPacman']
        if args['posGhost'] is None:
            self.posGhost = (((self.posPacman-1+randint(1, self.mazeLength-1)))%self.mazeLength) + 1
        else:
            self.posGhost = args['posGhost']
        self.listFood = []
        self.listCapsule = []
        for k in range(1,self.mazeLength+1): #randomization of food and capsules
            if k == self.posPacman or k == self.posGhost:
                continue
            #random.seed(0)
            randomInt = randint(0,2)
            if randomInt == 1:
               self.listCapsule.append(k)
            elif randomInt == 2:
               self.listFood.append(k)



class Rule:
    def __init__(self, closure):
        self.func = closure
    def check(self, gameData):
        return self.func(gameData)

def checkRules(rules, gameData, chromosome):
    for rule in rules:
        if rules[rule].check(gameData) is not chromosome[rule]:
            return False
    return True

# TODO
# Rewrite rules with binary relations and GP(Genetic Programming)
# Perhaps define another class GP? 
# with Objects = pacman, closestGhost, closestFood, closestCapsule, ...
# and Relations = isNear, atEast, atCorner, ... 

def isNear(pos1, pos2, near):
    if (abs(pos1 - pos2)<= near):
        return True
    return False

def atEast(pos1, pos2):
    if (pos1 > pos2):
        return True
    return False
    
def atCorner(pos):
    if(pos == 1 or pos == length):
        return True
    return False

def pacmanAtCorner(gameData):
    return atCorner(gameData.posPacman, gameData.mazeLength)

def ghostIsNear(gameData):
    return isNear( gameData.posGhost, gameData.posPacman, near = 1 )

def ghostAtEast(gameData):
    return atEast(gameData.posGhost, gameData.posPacman)

def ghostAtCorner(gameData):
    return atCorner(gameData.posGhost, gameData.mazeLength)

def closestFoodIsNear(gameData):
    #TODO
    # use closestFood at featureExtractors.py:29
    util.raiseNotDefined()

def closestFoodAtEast(gameData):
    #TODO
    util.raiseNotDefined()

def closestFoodAtCorner(gameData):
    #TODO
    util.raiseNotDefined()

def closestCapsuleIsNear(gameData):
    #TODO
    util.raiseNotDefined()

def closestCapsuleAtEast(gameData):
    #TODO
    util.raiseNotDefined()

def closestCapsuleAtCorner(gameData):
    #TODO
    util.raiseNotDefined()

#######################################################################
def foodIsNear(gameData, near = 1):
    for food in gameData.listFood:
        if(abs(gameData.posPacman - food) <= near):
            return True
    return False

def foodAtEast(gameData):
    for food in gameData.listFood:
        if(gameData.posPacman < food):
            return True
    return False
    
def capsuleIsNear(gameData, near = 1):
    for capsule in gameData.listCapsule:
        if(abs(gameData.posPacman - capsule) <= near):
            return True
    return False

def capsuleAtEast(gameData):
    for capsule in gameData.listCapsule:
        if(gameData.posPacman < capsule):
            return True
    return False
#######################################################################
