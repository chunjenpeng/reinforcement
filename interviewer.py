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

class Rule:
    def __init__(self, closure):
        self.func = closure
    def check(self, gameData):
        return self.func(gameData)

def ghostIsNear(gameData, near = 1):
    if (abs(gameData['posPacman'] - gameData['posGhost']) <= near):
        return True
    return False 

def ghostAtRight(gameData):
    if(gameData['posPacman'] < gameData['posGhost']):
        return True
    return False

def ghostAtLeft(gameData):
    if(gameData['posPacman'] > gameData['posGhost']):
        return True
    return False


listFood = []
listCapsule = []
seed = 0

usedGameStates = []
    
def ghostRule(gameData):
    rules = {} 
    rules['ghostIsNear'] = Rule(ghostIsNear)
    rules['ghostAtRight'] = Rule(ghostAtRight)
    rules['ghostAtLeft'] = Rule(ghostAtLeft)

    for rule in rules:
        print rule, rules[rule].check(gameData)
        if rules[rule].check is False:
            print 'Not '+rule
            return False
    return True

def generateLayout(gameData):
    listFood = []
    listCapsule = []
    height = gameData['mazeHeight']
    length = gameData['mazeLength']
    posPacman = gameData['posPacman']
    posGhost = gameData['posGhost']
    layoutText = [None]*(2+height)

    wall = "%"*(length+2)

    layoutText[0] = wall
    layoutText[height+1] = wall

    for k in range(1,length+1): #randomization of food and capsules
        if k == posPacman:
            continue
        if k == posGhost:
            continue
        #random.seed(0)
        randomInt = randint(0,2)
        if randomInt == 1:
            listCapsule.append(k)
        elif randomInt == 2:
            listFood.append(k)

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

def generateGameState(gameData): 
    layout =  generateLayout(gameData)
    gameState = GameState()
    numGhostAgents = 1
    gameState.initialize(layout, numGhostAgents)
    return gameState
    
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
                      help = default('the length of the maze'), default = 5)
    parser.add_option('--mazeHeight', dest = 'mazeHeight', type='int',
                      help = default('the height of the maze'), default = 1)
    parser.add_option('--posPacman', dest = 'posPacman', type='int',
                      help = default('the position of pacman in a horizontal maze'), default = 2)
    parser.add_option('--posGhost', dest = 'posGhost', type='int',
                      help = default('the position of the ghost in a horizontal maze'), default = 3)
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
    ghostRule(args)    
    gameState = generateGameState(args)
    print gameState
    print 'Pacman Action: ' + getAction(gameState) + "\n"
