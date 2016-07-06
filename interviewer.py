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
from collections import defaultdict
from itertools import product
from numpy import unravel_index
import featureGenerator
import numpy as np
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
        self.mazeHeight = len(layout)-2
        self.mazeLength = len(layout[0])-2
        self.listFood = []
        self.listCapsule = []

        pacman = 0
        for k in range(1, len(layout[1])):
            if layout[1][k] == "P":
                pacman = k
        self.posPacman = pacman

        ghost = 0
        for k in range(1, len(layout[1])):
            if layout[1][k] == "G":
                ghost = k
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

def closestWallIsNear(gameData, near = 1):
    return atCorner(gameData.posPacman, gameData.mazeLength)

def closestWallAtEast(gameData):
    if(gameData.posPacman >= gameData.mazeLength/2):
        return True
    else:
        return False

############# Remove this part when featureGenerator.py is finished #######################

def generateChromosome(chromosomeString):
    chromosome = dict.fromkeys(features.keys(),False)
    keys = chromosome.keys()
    for k in range (0,len(keys)):
        if chromosomeString[k] == '0':
            chromosome[keys[k]] = False
        else:
            chromosome[keys[k]] = True
    return chromosome

def generateChromosomes(chromosomeString):
    population = []
    #chromosomeString = '1**1***'
    c = chromosomeString.count('*')
    chromosomeString = chromosomeString.replace('*', '{}')
    for x in product('01', repeat=c):
        #print chromosomeString.format(*x)
        chromosomeStringInstance = chromosomeString.format(*x)
        #print 'chromosomeString',chromosomeString
        #print 'chromosomeStringInstance', chromosomeStringInstance
        chromosome = generateChromosome( chromosomeStringInstance )
        population.append(chromosome)
    return population
'''
def generateAllChromosomes(chromosomeNumber):
    allChromosomes = []
    for k in range(0, 2**(chromosomeNumber)):
        binaryvalue = str('{0:07b}'.format(k))
        allChromosomes.append(generateChromosome(binaryvalue))
    return allChromosomes
'''
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

    for k in allValidStates: #hardcoded for only ONE GHOST!!
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
        layouttext.append("%"*(length+2)) #HARDCODE
        layouttext.append(newstate)
        layouttext.append("%"*(length+2)) #HARDCODE
        allLayouts[k] = layouttext
        #print layouttext

    allStates = []
    for k in range(0, len(allLayouts)):
        layout = Layout(allLayouts[k])
        gameState = GameState()
        gameState.initialize(layout, 1) #ghost hardcoded
        allStates.append(gameState)
    return allStates



def findChromosomesStates(population, allStates, args):
    if len(population) <= 0:
        return
    features = generateFeatures()
    chromosomesWithStates = {} # {chromosome:[matchState1, matchState2, ...], ch2:[], ...} 
    for state in allStates:
        data = gameData(args)
        data.initializeWithState(state)
        for chromosome in population:
            if(satisfyFeatures(features, data, chromosome)):
                chromosomeString = chromosome2string(chromosome)
                chromosomesWithStates.setdefault(chromosomeString,[]).append(state)
                '''
                print ''
                print getFeatures(chromosome)
                print state
                print 'posPacman = ', str(data.posPacman)
                print 'posGhost = ', str(data.posGhost)
                print 'listFood = ', data.listFood
                print 'listCapsule = ', data.listCapsule
                '''
    return chromosomesWithStates            

def chromosome2string(chromosome):
    binarystring = ""
    for x in chromosome.values():
        if x:
            binarystring+= '1'
        else:
            binarystring+= '0'
    return binarystring

def printChromosomeList(chromosomes):
    for chromosome in chromosomes:
        print chromosome2string(chromosome)

def generateFeatures():
    features = {}
    features['ghostIsNear'] = Feature(ghostIsNear)
    features['ghostAtEast'] = Feature(ghostAtEast)
    features['closestFoodIsNear'] = Feature(closestFoodIsNear)
    features['closestFoodAtEast'] = Feature(closestFoodAtEast)
    features['closestCapsuleIsNear'] = Feature(closestCapsuleIsNear)
    features['closestCapsuleAtEast'] = Feature(closestCapsuleAtEast)
    features['closestWallIsNear'] = Feature(closestWallIsNear)
    features['closestWallAtEast'] = Feature(closestWallAtEast)
    #features['pacmanAtCorner'] = Feature(pacmanAtCorner) 
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

def printContradictRules(badChromosomes):
    print 'Contradict rules:' + str(len(badChromosomes))
    for chromosome in badChromosomes:
        print getFeatures(chromosome)

def chromosome2bit(chromosome):
    bitList = []
    for x in chromosome.values():
        if x:
            bitList.append(1)
        else:
            bitList.append(0)
    return bitList 

def testChromosomes(chromosomesWithStates):
    badChromosomes = []
    goEastChromosomes = []
    goWestChromosomes = []
    for chromosomeString in chromosomesWithStates.keys():
        chromosome = generateChromosome(chromosomeString)
        goEast = goWest = 0.0 
        stateList = chromosomesWithStates[chromosomeString]
        if len(stateList) == 0:
            badChromosomes.append(chromosome)
            continue
        for gameState in stateList:
            action = getAction(gameState)
            #print gameState, action
            if action == 'West':
                goWest = goWest + 1.0
            if action == 'East':
                goEast = goEast + 1.0
        
        if goEast/len(stateList) >= successRate:
            #print 'go East', str(100*goEast/len(stateList)), '% :', getFeatures(chromosome) 
            goEastChromosomes.append(chromosome)
        if goWest/len(stateList) >= successRate:
            #print 'go West', str(100*goWest/len(stateList)), '% :', getFeatures(chromosome)
            goWestChromosomes.append(chromosome)

    return badChromosomes, goEastChromosomes, goWestChromosomes


def findMI(chromosomes):
    sum_list = [sum(x) for x in zip(*chromosomes)]
    p_list = [float(sum(x))/len(chromosomes) for x in zip(*chromosomes)]
    print sum_list, p_list

def calc_MI(X,Y,bins):
    c_XY = np.histogram2d(X,Y,bins)[0]
    c_X = np.histogram(X,bins)[0]
    c_Y = np.histogram(Y,bins)[0]

    H_X = shan_entropy(c_X)
    H_Y = shan_entropy(c_Y)
    H_XY = shan_entropy(c_XY)

    MI = H_X + H_Y - H_XY
    return MI

def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log2(c_normalized))
    return H

def calc_matMI(A):
    n = A.shape[1]
    bins = n  
    matMI = np.zeros((n,n))
    for ix in np.arange(n):
        for jx in np.arange(ix+1, n):
            matMI[ix, jx] = calc_MI(A[:, ix], A[:,jx], bins)
            matMI[jx, ix] = calc_MI(A[:, ix], A[:,jx], bins)
    return matMI
'''
def findMask(A):
    featureList = []
    feature1, feature2 = unravel_index(A.argmax(), A.shape)
    featureList.append(feature1)
    featureList.append(feature2)
    n = A.shape[1]
    Array = np.zeros((n,n))
    Array[:] = A
    print '(',feature1,',',feature2,')',Array[feature1][feature2]
    print Array
    Array[feature1][feature2] = 0
    
    for i in range(0, A.shape[0]-2):
        if (np.amax(Array[feature1,:]) > np.amax(Array[feature2,:])):
            print '(',feature1,',',np.nanargmax(Array[feature1,:]),') =',np.amax(Array[feature1,:])
            Array[feature1][feature2] = 0
            feature1 = np.nanargmax(Array[feature1,:])
            featureList.append(feature1)
            print Array 
        else:
            print '(',feature2,',',np.nanargmax(Array[feature2,:]),') =',np.amax(Array[feature2,:])
            Array[feature1][feature2] = 0
            feature2 = np.nanargmax(Array[feature2,:])
            featureList.append(feature2)
            np.set_printoptions(suppress=True, precision=3)
            print Array 
    
    return featureList
'''

def findMask(A,startNode):
    featureList = []
    feature1 = startNode
    feature2 = np.nanargmax(A[feature1,:])
    featureList.append(feature1)
    featureList.append(feature2)
    n = A.shape[1]
    Array = np.zeros((n,n))
    Array[:] = A
    #print '(',feature1,',',feature2,') =',Array[feature1][feature2]
    #print Array
    Array[feature1][feature2] = 0
    Array[feature2][feature1] = 0
    
    for i in range(0, A.shape[0]-2):
        if (np.amax(Array[feature1,:]) > np.amax(Array[feature2,:])):
            #print '(',feature1,',',np.nanargmax(Array[feature1,:]),') =',np.amax(Array[feature1,:])
            Array[feature1][feature2] = 0
            Array[feature2][feature1] = 0
            feature1 = np.nanargmax(Array[feature1,:])
            if feature1 in featureList:
                break
            else:
                featureList.append(feature1)
                #print Array 
        else:
            #print '(',feature2,',',np.nanargmax(Array[feature2,:]),') =',np.amax(Array[feature2,:])
            Array[feature1][feature2] = 0
            Array[feature2][feature1] = 0
            feature2 = np.nanargmax(Array[feature2,:])
            if feature2 in featureList:
                break
            else:
                featureList.append(feature2)
                #print Array 
    return featureList


def calc_2bits_prob(A):
    n00 = n01 = n10 = n11 = 0.0
    for f in A:
        if f[0] == 0 and f[1] == 0:
            n00 = n00 + 1
        elif f[0] == 0 and f[1] == 1:
            n01 = n01 + 1
        elif f[0] == 1 and f[1] == 0:
            n10 = n10 + 1
        elif f[0] == 1 and f[1] == 1:
            n11 = n11 + 1
    print 'n00', n00, n00/len(A)
    print 'n01', n01, n01/len(A)
    print 'n10', n10, n10/len(A)
    print 'n11', n11, n11/len(A)
    nList = [n00/len(A), n01/len(A), n10/len(A), n11/len(A)]

def findPossibleString(mask, arr, baseString):
    stringDict = {} 
    dictFeature = {}
    possible = list(baseString) 
   
    matFeature = arr[:, mask]
    for i in xrange(len(matFeature)):
        key = ''.join(map(str,matFeature[i]))
        if key in dictFeature:
            dictFeature[key] += 1
        else:
            dictFeature[key] = 1
    #print arr
    print '\nUsing mask:',mask
    print dictFeature
    while dictFeature: 
        s = max(dictFeature, key=dictFeature.get)
        for i in xrange(len(mask)):
            possible[mask[i]] = s[i]
        possibleString = ''.join(possible)
        stringDict[possibleString] = dictFeature[s]
        #print 'possibleString', possibleString, ',score = ', dictFeature[s]
        del dictFeature[s] 
    return stringDict

def findRelatedFeatures(Array):
    n = Array.shape[1]
    A = np.zeros((n,n))
    A[:] = Array
    relatedFeatureList = []
    feature1, feature2 = unravel_index(A.argmax(), A.shape)
    while feature1 != feature2:
        pair = (feature1, feature2)
        relatedFeatureList.append(pair)
        A[feature1][feature2] = 0
        A[feature2][feature1] = 0
        feature1, feature2 = unravel_index(A.argmax(), A.shape)
    return relatedFeatureList

def mergeFeatures(chromosomes, baseString):
    if chromosomes is None:
        return
    bitLists = []
    for chromosome in chromosomes:
        bitChromosome = chromosome2bit(chromosome)
        bitLists.append(bitChromosome)
        #print bitChromosome#, getFeatures(chromosome)
    arr = np.array(bitLists)
    print 'Mutual Information Matrix:'
    matMI = calc_matMI(arr)                                 # Mutual Information Matrix
    np.set_printoptions(suppress=True, precision=3)
    print matMI
    
    print 'Related features:' 
    relatedFeatureList = findRelatedFeatures(matMI)         # Related Feature List
    keys = list(features.keys())
    for feature1, feature2 in relatedFeatureList:
        print '(',feature1,',',feature2,') = ', matMI[feature1][feature2], keys[feature1], keys[feature2]
    
    masks = []                                              # Masks
    for i in range(0, len(chromosomes[0])):
        mask = findMask(matMI, i) # ILS 
        masks.append(mask)
    #    print mask
    print 'masks :', masks 
    
    stringDict = {}
    #TODO
    usedFeature = []
    for i in xrange(len(relatedFeatureList)):
        feature1 = relatedFeatureList[i][0]
        if feature1 in usedFeature:
            continue
        usedFeature.append(feature1)
        possibleStringDict = findPossibleString(masks[feature1], arr, baseString)
        print possibleStringDict
        stringDict.update( possibleStringDict )
    return stringDict

def findLearnedFeatures(string):
    learnedFeature = [] 
    learnedStringList = ['*'] * len(features)
    slist = list(string)
    for s in range(0,len(slist)):
        if slist[s] != '*':
            learnedStringList[s] = slist[s]
    #print 'learned string:', ''.join(learnedStringList) 
    learnedString = str(''.join(learnedStringList))
    
    s = learnedStringList
    keys = list(features.keys())
    for feature in xrange(len(learnedStringList)):
        if s[feature] == '0':
            learnedFeature.append('Not'+keys[feature])
        elif s[feature] == '1':
            learnedFeature.append(keys[feature])
    #print 'Learned Features: ', ' and '.join(learnedFeature)
    return learnedFeature

def selection(population, condition): #condition is a string '1*1****'
    newPopulation = []
    newChromosomes= generateChromosomes(condition) 
    for newChromosome in newChromosomes:
        if newChromosome in population:
            newPopulation.append(newChromosome)
    return newPopulation

############################################################################################################

args = readCommand( sys.argv[1:] )
features = generateFeatures()
successRate = 0.7
learnedFeatures = {} #{learnedFeature:score} 
learnedStrings = {} #{learnedString:score} 
initialString = '*' * len(features)
population = generateChromosomes(initialString)
populationDict = {}
allStates = generateAllStates(args["mazeLength"])
TERMINATE = False 

#while not TERMINATE: 

populationDict[initialString]=population
chromosomesWithStates = findChromosomesStates(population, allStates, args)
badChromosomes, goEastChromosomes, goWestChromosomes = testChromosomes(chromosomesWithStates)


#Check goEastChromosomes
print'\nOver', str(100*successRate), '% of the time, Pacman goes East: '+str(len(goEastChromosomes))
stringDict = {}
stringDict = mergeFeatures(goEastChromosomes, initialString)
#print 'stringDict', stringDict
learnedStrings.update(stringDict) 
learnedStringsSorted = sorted (((learnedString, score) for score, learnedString in learnedStrings.iteritems()), reverse=True)

print '\n\nfeatures: ', ', '.join(features.keys())
print '\nLearned Features for Action East:'
for score, learnedString in learnedStringsSorted:
    learnedFeature = findLearnedFeatures(learnedString)
    print 'score:', score, learnedString, 'When', ' and '.join(learnedFeature)
    
    '''
    #Check goWestChromosomes
    print'\nOver', str(100*successRate), '% of the time, Pacman goes West: '+str(len(goEastChromosomes))
    stringDict = {}
    stringDict = mergeFeatures(goWestChromosomes, initialString)
    print 'stringDict', stringDict
    learnedStrings.update(stringDict) 
    '''


    #if stringList[0].count('*') == len(features) - 1:
    #    break 
    
#pause = raw_input("\nPress <ENTER> to continue...\n\n")



''' 
    for learnedString in stringDict:
        print 'learned string:', learnedString 
        #learnedStrings[learnedString] = stringDict[learnedString] # stringDict[learnedString] = score
        learnedFeature = findLearnedFeatures(learnedString)
        #learnedFeatures.append(learnedFeature)
        print 'Learned Feature: ', ' and '.join(learnedFeature)
        
        
        newPopulation = selection(population, learnedString) 
        populationDict[learnedString] = newPopulation
        #print populationDict[learnedString] 
        chromosomesWithStates = findChromosomesStates(populationDict[learnedString], allStates, args)
        badChromosomes, goEastChromosomes, goWestChromosomes = testChromosomes(chromosomesWithStates)
        print'Over', str(100*successRate), '% of the time, Pacman goes East: '+str(len(goEastChromosomes))
        
        stringDict = mergeFeatures(goEastChromosomes, learnedString)
        print 'stringDict', stringDict
        learnedStrings.update(stringDict) 
      
        pause = raw_input("\nPress <ENTER> to continue...\n\n")
        
        ############################################################################
        for learnedString in stringDict.keys():
            learnedFeature = findLearnedFeatures(learnedString)
            learnedFeatures.append(learnedFeature)
            learnedStrings.append(learnedString)
            print 'learned string:', learnedString 
            print 'Learned Features: ', ' and '.join(learnedFeature)
            
            populationDict[learnedString] = selection(population, learnedString) 
            #print populationDict[learnedString] 
            chromosomesWithStates = findChromosomesStates(populationDict[learnedString], allStates, args)
            badChromosomes, goEastChromosomes, goWestChromosomes = testChromosomes(chromosomesWithStates)
            print'Over', str(100*successRate), '% of the time, Pacman goes East: '+str(len(goEastChromosomes))
            
            stringDict = mergeFeatures(goEastChromosomes, learnedString)
    
            pause = raw_input("Press <ENTER> to continue...")
        ############################################################################
        '''












'''
print'\nPacman goes East when: '
for chromosome in goEastChromosomes:
    print generateGameState(gameData(args, chromosome))
    print getFeatures(chromosome)
print'\nPacman goes West when: '
for chromosome in goWestChromosomes:
    print generateGameState(gameData(args, chromosome))
    print getFeatures(chromosome)
'''



'''
bitLists = []
print'\nOver', str(100*successRate), '% of the time, Pacman goes West: '+str(len(goWestChromosomes))
if len(goWestChromosomes) > 0:
    for chromosome in goWestChromosomes:
        bitChromosome = chromosome2bit(chromosome)
        bitLists.append(bitChromosome)
        print bitChromosome#, getFeatures(chromosome)
    print 'Mutual Information Matrix:'
    matMI = calc_matMI(np.array(bitLists))
    print matMI
    feature1, feature2 =  unravel_index(matMI.argmax(), matMI.shape)
    keys = list(features.keys())
    print 'Most related features:', keys[feature1], keys[feature2] 
'''


#printChromosomeList(goWestChromosomes)

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
