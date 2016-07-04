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
import random

#TODO
############# Remove this part when featureGenerator.py is finished #######################
class gameData:
    def __init__(self, args, chromosome = None):
        self.initialize(args)
        if chromosome != None:
            while satisfyFeatures(generateFeatures(), self, chromosome) == False:
                self.initialize(args)
    def initialize(self, args):
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
    def initializeWithState(self, gameState):
        layout = gameState.data.layout.layoutText
        self.mazeHeight = len(layout)
        self.mazeLength = len(layout[0])

        pacman = 0
        for k in range(1, len(layout[1])):
            if layout[1][k] == "P":
                pacman = k-1
        self.posPacman = pacman

        ghost = 0
        for k in range(1, len(layout[1])):
            if layout[1][k] == "P":
                ghost = k-1
        self.posGhost = ghost

        for k in range(1, len(layout[1])):
            if layout[1][k] == ".":
                self.listFood.append(k)

        for k in range(1, len(layout[1])):
            if layout[1][k] == "o":
                self.listCapsule.append(k)
        return self



class Feature:
    def __init__(self, closure):
        self.func = closure
    def satisfy(self, gameData):
        return self.func(gameData)

def satisfyFeatures(feature, gameData, chromosome):
    for feature in features:
        if features[feature].satisfy(gameData) is not chromosome[feature]:
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

def closestFoodIsNear(gameData, near = 1):
    return util.closest(gameData.listFood,args['mazeLength'],gameData) <= near

def closestFoodAtEast(gameData):
    closestList = util.closestList(gameData.listFood,args['mazeLength'],gameData)
    if (len(closestList) == 0):
        return False
    if (len(closestList) == 1):
        if (gameData.posPacman < closestList[0]):
            return True
        return False
    if (len(closestList) == 2):
        return True
    
def closestCapsuleIsNear(gameData, near = 1):
    return util.closest(gameData.listCapsule, args['mazeLength'], gameData) == near

def closestCapsuleAtEast(gameData):
    closestList = util.closestList(gameData.listCapsule,args['mazeLength'],gameData)
    if(len(closestList)==0):
        return False
    if(len(closestList)==1):
        if(gameData.posPacman < closestList[0]):
            return True
        return False
    if(len(closestList)==2):
        return True
############# Remove this part when featureGenerator.py is finished #######################

def generateChromosome(chromosomeString = '0000000'):
    chromosome = dict.fromkeys(features.keys(),False)
    keys = chromosome.keys()
    for k in range (0,len(keys)):
        if chromosomeString[k] == '0':
            chromosome[keys[k]] = False
        else:
            chromosome[keys[k]] = True
    return chromosome

def generateAllChromosomes(chromosomeNumber):
    allChromosomes = []
    for k in range(0, 2**(chromosomeNumber)):
        binaryvalue = str('{0:07b}'.format(k))
        allChromosomes.append(generateChromosome(binaryvalue))
    return allChromosomes

def generateAllStates(length, ghostNum = 1): #length of all possible spaces. Do not set the ghost num
    allStatesWithoutP = []
    for k in range(0, 4**length):
        layout = util.base10toN(k, 4, length)
        allStatesWithoutP.append(layout)

    allValidStates = []

    for k in allStatesWithoutP: 
        zerocount = 0
        for x in range(0, len(k)):
            if k[x] == "0":
                zerocount += 1
        if zerocount == (ghostNum+1):
            allValidStates.append(k)

    allLayouts = []

    for k in allValidStates: #hardcoded because I couldn't think of a better way of doing this
        tempstring1 = ""
        tempstring2 = ""
        switcher = True
        for x in range(0, len(k)):
            if k[x] == "0":
                if switcher:
                    tempstring1 += "4"
                    tempstring2 += "5"
                else:
                    tempstring1 += "5"
                    tempstring2 += "4"
                switcher = False
            else:
                tempstring1 += k[x]
                tempstring2 += k[x]
        allLayouts.append(tempstring1)
        allLayouts.append(tempstring2)
    for k in range(0, len(allLayouts)):
        state = allLayouts[k]
        newstate = "%"
        for x in range(0, len(state)):
            if state[x] == "1":
                newstate+=" "
            elif state[x] == "2":
                newstate += "."
            elif state[x] == "3":
                newstate+= "o"
            elif state[x] == "4":
                newstate+= "P"
            elif state[x] == "5":
                newstate+= "G"
        newstate+= "%"
        layouttext = []
        layouttext.append("%"*7) #HARDCODE
        layouttext.append(newstate)
        layouttext.append("%"*7) #HARDCODE
        allLayouts[k] = layouttext
        print layouttext

    allStates = []
    for k in range(0, len(allLayouts)):
        layout = Layout(allLayouts[k])
        gameState = GameState()
        gameState.initialize(layout, 1) #ghost hardcoded
        allStates.append(gameState)
    return allStates



def testChromosomes(chromosomeNumber, args, testlimit):
    allChromosomes = generateAllChromosomes(chromosomeNumber)
    allStates = generateAllStates(args["mazeLength"])
    allData =[]
    for state in allStates:
        data = gameData(args)
        allData.append(data.initializeWithState(state))
    features = generateFeatures()
    badChromosomes = []
    goodChromosomes = []
    for k in allChromosomes:
        usedData = []
        for data in allData:
            print data
            if(satisfyFeatures(features, data, k)):
                usedData.append(data)
        if len(usedData) == 0:
            badChromosomes.append(k)
        else:
            goodChromosomes.append(k)
    #print "The contradictory chromosomes are:"
    #printChromosomeList(badChromosomes)
    return goodChromosomes, badChromosomes

def printChromosome(chromosome):
    binarystring = ""
    for x in chromosome.values():
        if x:
            binarystring+= '1'
        else:
            binarystring+= '0'
    print binarystring
def printChromosomeList(chromosomes):
    for k in chromosomes:
        printChromosome(k)

def generateFeatures():
    features = {}
    features['ghostIsNear'] = Feature(ghostIsNear)
    features['ghostAtEast'] = Feature(ghostAtEast)
    features['closestFoodIsNear'] = Feature(closestFoodIsNear)
    features['closestFoodAtEast'] = Feature(closestFoodAtEast)
    features['closestCapsuleIsNear'] = Feature(closestCapsuleIsNear)
    features['closestCapsuleAtEast'] = Feature(closestCapsuleAtEast)
    features['pacmanAtCorner'] = Feature(pacmanAtCorner)
    return features
    
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

def generateGameState(gameData): 
    layout =  generateLayout(gameData)
    gameState = GameState()
    numGhostAgents = 1
    gameState.initialize(layout, numGhostAgents)
    return gameState
    

def getAction(gameState):
    import pacmanAgents, qlearningAgents
    pacmanAgent = pacmanAgents.GreedyAgent()
    action = pacmanAgent.getAction(gameState)
    return action

def default(string):
  return string + ' [Default: %default]'

def readCommand(argv):

    from optparse import OptionParser

    usageStr = ""
    parser = OptionParser(usageStr)

    parser.add_option('--mazeLength', dest = 'mazeLength', type='int',
                      help = default('the length of the maze'), default = 5)
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

def getFeatures(chromosome):
    fullFeature = ''
    for feature in features:
        if chromosome[feature] is False: 
            fullFeature = fullFeature + 'Not'
        fullFeature = fullFeature + str(feature)+', ' 
    return fullFeature

args = readCommand( sys.argv[1:] )
features = generateFeatures()

goodChromosomes, badChromosomes = testChromosomes(len(generateChromosome()), args, 1000)
print 'Contradict rules:' + str(len(badChromosomes))
for chromosome in badChromosomes:
    print getFeatures(chromosome)

goEastChromosomes = []
goWestChromosomes = []
repeat = 10
successRate = 0.9
for chromosome in goodChromosomes:
    goEast = goWest = 0 
    #print ''
    #print getFeatures(chromosome) 
    for i in range(0,repeat):
        gameState = generateGameState(gameData(args, chromosome))
        action = getAction(gameState)
        if action == 'West':
            goWest = goWest+1
        if action == 'East':
            goEast = goEast+1
        #print gameState
        #print action
        #print 'goEast: '+str(goEast)+', goWest: '+str(goWest)
    if goEast >= repeat*successRate:
        goEastChromosomes.append(chromosome)
    if goWest >= repeat*successRate:
        goWestChromosomes.append(chromosome)

print'\nPacman goes East when: '
for chromosome in goEastChromosomes:
    print generateGameState(gameData(args, chromosome))
    print getFeatures(chromosome)
print'\nPacman goes West when: '
for chromosome in goWestChromosomes:
    print generateGameState(gameData(args, chromosome))
    print getFeatures(chromosome)

print'\nPacman goes East: '+str(len(goEastChromosomes))
printChromosomeList(goEastChromosomes)
print'\nPacman goes West: '+str(len(goWestChromosomes))
printChromosomeList(goWestChromosomes)

'''for k in range(0,args['numLayouts']):
    data = gameData(args)
    features = generateFeatures()
    # TODO
    # randomly generate chromosome until all(most) of the chromosome(features) have the same action
    # then extract the same features in the chromosome and combine them as a new feature
    chromosome = generateChromosome()
    data = gameData(args, chromosome)
    gameState = generateGameState(data)
    print gameState

    fullFeature = ""
    for feature in features:
        if not chromosome[feature]: 
            fullFeature = fullFeature + 'Not'
        fullFeature = fullFeature + str(feature)+', ' 

    print 'When: '+fullFeature+'Pacman goes: ' + getAction(gameState) + "\n"'''

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
