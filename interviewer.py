#author Daniel & Ryan
import util, sys
from layout import Layout
from pacman import GameState
from game import GameStateData
from game import Game
from game import Directions
from game import Actions
from random import randint
from random import seed
import featureGenerator

#TODO
############# Remove this part when featureGenerator.py is finished #######################
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

def isNear(pos1, pos2, near):
    if (abs(pos1 - pos2)<= near):
        return True
    return False

def atEast(pos1, pos2):
    if (pos1 > pos2):
        return True
    return False
    
def atCorner(pos, length):
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
############# Remove this part when featureGenerator.py is finished #######################

def generateChromosome():
    chromosome = dict.fromkeys(rules.keys(),False)
    #chromosome['ghostIsNear'] = True 
    #chromosome['ghostAtEast'] = True 
    chromosome['foodIsNear'] = True 
    chromosome['foodAtEast'] = True 
    #chromosome['capsuleIsNear'] = True 
    #chromosome['capsuleAtEast'] = True 
    #chromosome['pacmanAtCorner'] = True 
    return chromosome

def generateRules():
    rules = {}
    rules['ghostIsNear'] = Rule(ghostIsNear)
    rules['ghostAtEast'] = Rule(ghostAtEast)
    rules['foodIsNear'] = Rule(foodIsNear)
    rules['foodAtEast'] = Rule(foodAtEast)
    rules['capsuleIsNear'] = Rule(capsuleIsNear)
    rules['capsuleAtEast'] = Rule(capsuleAtEast)
    rules['pacmanAtCorner'] = Rule(pacmanAtCorner)
    return rules
    
def generateLayout(gameData):
    height = gameData.mazeHeight
    length = gameData.mazeLength
    posPacman = gameData.posPacman 
    posGhost = gameData.posGhost 
    listFood = gameData.listFood
    listCapsule = gameData.listCapsule

    layoutText = [None]*(2+height)
    wall = "%"*(length+2)
    layoutText[0] = wall
    layoutText[height+1] = wall

    for x in range(1,(height+1)):
        row = "%"

        for k in range(1,(length+1)):
            if k == posPacman:
                row += "P"
            elif k == posGhost:
                row += "G"
            elif k in listFood:
                row += "."
            elif k in listCapsule:
                row += "o"
            else:
                row += " "

        row += "%"
        layoutText[x] = row

    return Layout(layoutText)

def generateGameState(args): 
    layout =  generateLayout(args)
    gameState = GameState()
    numGhostAgents = 1
    gameState.initialize(layout, numGhostAgents)
    return gameState
    

def getAction(gameState):
    import pacmanAgents, qlearningAgents
    pacmanAgent = pacmanAgents.GreedyAgent()
    action = pacmanAgent.getAction(gameState)
    return action

def default(str):
  return str + ' [Default: %default]'

def readCommand(argv):

    from optparse import OptionParser

    usageStr = ""
    parser = OptionParser(usageStr)

    parser.add_option('--mazeLength', dest = 'mazeLength', type='int',
                      help = default('the length of the maze'), default = 10)
    parser.add_option('--mazeHeight', dest = 'mazeHeight', type='int',
                      help = default('the height of the maze'), default = 1)
    parser.add_option('--posPacman', dest = 'posPacman', type='int',
                      help = default('the position of pacman in a horizontal maze'), default = None)
    parser.add_option('--posGhost', dest = 'posGhost', type='int',
                      help = default('the position of the ghost in a horizontal maze'), default = None)
    parser.add_option('--numLayouts' ,dest = 'numLayouts', type='int',
                      help = default('the number of layouts to be generated'), default = 10 )

    options, otherjunk = parser.parse_args(argv)

    if len(otherjunk) != 0:
        raise Exception('Command line input not understood: ' + str(otherjunk))

    args = dict()

    args['mazeLength'] = options.mazeLength
    args['mazeHeight'] = options.mazeHeight
    args['posPacman'] = options.posPacman
    args['posGhost'] = options.posGhost
    args['numLayouts'] = options.numLayouts
    return args


args = readCommand( sys.argv[1:] )
for k in range(0,args['numLayouts']):
    data = gameData(args)
    rules = generateRules()
    # TODO
    # randomly generate chromosome until all(most) of the chromosome(rules) have the same action
    # then extract the same rules in the chromosome and combine them as a new rule
    chromosome = generateChromosome()
    while checkRules(rules, data, chromosome) is False:
        #TODO 
        # add data into a usedGameData dict
        # if usedGameData has all possible gamestates, return that rules contain conflict rules
        data = gameData(args)
    gameState = generateGameState(data)
    print gameState

    fullRule = ""
    for rule in rules:
        if not chromosome[rule]: 
            fullRule = fullRule + 'Not'
        fullRule = fullRule + str(rule)+', ' 

    print 'When: '+fullRule+'Pacman goes: ' + getAction(gameState) + "\n"

'''
def generateGameStates(gameData):
    listGameStates = []
    randomSeed = 0
    for repeat in range (0, gameData['numLayouts']):
        if ghostRule(gameData):
            gameState = generateGameState(gameData)
            if gameState not in usedGameStates:
                listGameStates.append(gameState)
                usedGameStates.append(gameState)
    return listGameStates
'''
