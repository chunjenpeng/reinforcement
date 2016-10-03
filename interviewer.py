import json, random
from operator import itemgetter
from itertools import product
import cPickle as pickle

import featureGenerator as fg
import numpy as np
from numpy import unravel_index
from sklearn.metrics import normalized_mutual_info_score

def chromosome2feature(chromosome, features):
    feature_string = ''  
    for i in xrange(len(chromosome)):
        if chromosome[i] == '*': continue
        if i > 0:
            feature_string += ' and '
        if chromosome[i] == '0':
            feature_string += 'Not '

        feature_string += features[i].__str__()
    return feature_string


def generateDescendants(string):
    descendants = []
    for i in xrange(len(string)):
        if string[i] != '*':
            child = list(string)
            child[i] = '*'
            descendants.append(''.join(child))
    return descendants


def generateChromosomes(string):
    population = []
    c = string.count('*')
    string = string.replace('*', '{}')
    for x in product('01', repeat=c):
        chromosome = string.format(*x)
        population.append(chromosome)
    return population


def generateInitialPopulation(features, n): 
    initialString = '*' * len(features) 
    population = generateChromosomes(initialString) 
    if n >= len(population):
        return population
    random.shuffle(population)
    return population[:n]


def matchDataWithChromosome( chromosome, features, observations):
    chromosomeData = [] 
    satisfyState = {'North':[], 'South':[], 'East':[], 'West':[], 'Stop':[]}
     
    for data in observations:    # ??efficiency??
        if fg.satisfyFeatures( chromosome, features, data['gameState'] ):
            chromosomeData.append(data)
            for action in satisfyState:
                if action == data['action']:
                    satisfyState[action].append(1)
                else:
                    satisfyState[action].append(0)
        else:
            for action in satisfyState:
                satisfyState[action].append(0)
    
    return chromosomeData, satisfyState 


def calculateActionAccuracy(chromosomeData): #Accuracy of chromosome or Accuracy of gameStates?
    accuracyOfChromosome = {'North':0.0, 'South':0.0, 'East':0.0, 'West':0.0, 'Stop':0.0}
    for data in chromosomeData:
        action = data['action']
        accuracyOfChromosome[action] += 1.0
    data_num = len(chromosomeData)
    if data_num != 0:
        for action in accuracyOfChromosome:
            accuracyOfChromosome[action] /= data_num
    return accuracyOfChromosome 


def findBestAction(chromosome, features, observations):
    chromosomeData, satisfyState  = matchDataWithChromosome(chromosome, features, observations)
    
    allActionAccuracy = calculateActionAccuracy(chromosomeData)
    bestAction = max(allActionAccuracy, key=allActionAccuracy.get) 
    accuracy = allActionAccuracy[bestAction]
    #accuracy = allActionAccuracy[bestAction]*len(chromosomeData)/len(observations)
    
    satisfyStateList = satisfyState[bestAction] 
    bestData = {'bestAction':bestAction, 'accuracy':accuracy, 'satisfyStateList':satisfyStateList, 'chromosomeData':chromosomeData}
    return bestData


def steepestDescent(chromosome, features, observations, bestResponse):
    '''
    bestResponse = { 
        chromosome: {
            'bestAction':bestAction, 
            'accuracy':accuracy, 
            'satisfyStateList':satisfyStateList, 
            'chromosomeData':chromosomeData
        },  
        ...
    }
    ''' 
    bestChromosome = chromosome
    currentBestData = findBestAction(chromosome, features, observations) 
    bestResponse[chromosome] = currentBestData 
    
    descendants = generateDescendants(chromosome)
    for child in descendants:
        if child in bestResponse:
            nextBestData = bestResponse[child]
        else:
            nextBestData = findBestAction(child, features, observations) 
            bestResponse[child] = nextBestData 
        
        if nextBestData['accuracy'] >= currentBestData['accuracy']:
            bestChromosome = child 
            currentBestData = nextBestData
    
    return bestChromosome


def hillClimbing(population, features, observations, bestResponse, doPrint=False):
    nextGeneration= [] 
    if doPrint: print '\nSteepest Descent for each chromosome...'
    for chromosome in population:
        child = steepestDescent(chromosome, features, observations, bestResponse) 
        if doPrint: print chromosome,
        while(child != chromosome):
            if doPrint: print '->', child,
            chromosome = child
            child = steepestDescent(chromosome, features, observations, bestResponse) 
        if doPrint: print '' 
        
        if child not in nextGeneration: 
            nextGeneration.append(child)
    
    return nextGeneration

def calcResponseAccuracy(population, bestResponse, features):
    '''
    bestResponse = { 
        chromosome: {
            'bestAction':bestAction, 
            'accuracy':accuracy, 
            'satisfyStateList':satisfyStateList, 
            'chromosomeData':chromosomeData
        },  
        ...
    }
    ''' 
    responseWithAccuracy = [] 
    for chromosome in population:
        data = bestResponse[chromosome]
        response = (chromosome, data['bestAction']) 
        accuracy = data['accuracy']
        responseWithAccuracy.append((response, accuracy))
    
    responseWithAccuracy.sort(key=itemgetter(1), reverse=True)
    return responseWithAccuracy

def generateMatrix(A):
    n = A.shape[1]
    matMI = np.zeros((n,n))
    for ix in np.arange(n):
        for jx in np.arange(ix+1, n):
            matMI[ix, jx] = normalized_mutual_info_score(A[:, ix], A[:,jx])
            matMI[jx, ix] = normalized_mutual_info_score(A[:, jx], A[:,ix])
    return matMI

def findMutualInformation(population, bestResponse):
    matrix = []
    for chromosome in population:
        satisfyStateList = bestResponse[chromosome]['satisfyStateList'] 
        matrix.append(satisfyStateList)
    arr = np.array(matrix)
    #matMI = generateMatrix(np.transpose(arr))
    matMI = generateMatrix(arr.T)
    return matMI

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

def printResponse(chromosome, bestResponse, features):
    action = bestResponse[chromosome]['bestAction']
    accuracy = bestResponse[chromosome]['accuracy']
    dataNum = len(bestResponse[chromosome]['chromosomeData'])
    print ('%8.7f' % accuracy),('(%4d)' % dataNum), ':', chromosome, 'Go', action, ', when', chromosome2feature(chromosome, features)

def printKnowledge(knowledge, features):
    # knowledge = {'chromosomes':[ch1, ch2], 'action': action, 'data': data, 'accuracies':[ac1, ac2], 'MI':MI}
    print '\nKnowledge:'
    ch1, ch2 = knowledge['chromosomes']
    action = knowledge['action']
    print ch1, 'Go', action, ', when', chromosome2feature(ch1, features)
    print ch2, 'Go', action, ', when', chromosome2feature(ch2, features)
    print 'Number of states removed:', len(knowledge['data'])
    print 'accuracy:', knowledge['accuracies']
    print 'Mutual Information:', knowledge['MI']

def oneRun(observations, features, population):    
    # Record chromosome score
    bestResponse = {} 
    '''
    bestResponse = { 
        chromosome: {
            'bestAction':bestAction, 
            'accuracy':accuracy, 
            'satisfyStateList':satisfyStateList, 
            'chromosomeData':chromosomeData
        },  
        ...
    }
    ''' 
    
    print '\nNumber of gameStates:', len(observations)
    
    nextGeneration = hillClimbing(population, features, observations, bestResponse)
    responseWithAccuracy = calcResponseAccuracy(nextGeneration, bestResponse, features)
    sorted_nextGeneration = [chromosome for ((chromosome, action), accuracy) in responseWithAccuracy]
    
    print '\nResponse Accuracy:'
    for ((chromosome, action), accuracy) in responseWithAccuracy:
        printResponse(chromosome, bestResponse, features)
    
    if len(nextGeneration) == 1: 
        CONTINUE = False
        return CONTINUE, {}, observations, bestResponse, sorted_nextGeneration 
    
    matMI = findMutualInformation(sorted_nextGeneration, bestResponse)
    relatedFeatureList = findRelatedFeatures(matMI)
    #np.set_printoptions(precision=3, suppress=True)
    #print '\nMutual Information Matrix:\n', matMI
    
    print '\nRelated features:' 
    for i, j in relatedFeatureList:
        if matMI[i][j] < 0.99: break
        print 'Mutual Information(',i,',',j ,') = ', matMI[i][j]
        printResponse(sorted_nextGeneration[i], bestResponse, features)
        printResponse(sorted_nextGeneration[j], bestResponse, features)

    i, j = relatedFeatureList[0]
    ch1 = sorted_nextGeneration[i] 
    ch2 = sorted_nextGeneration[j] 
    sorted_nextGeneration.remove(ch1)
    sorted_nextGeneration.remove(ch2)
    action = bestResponse[ch1]['bestAction']
    data = []
    for d in bestResponse[ch1]['chromosomeData']:
        if d['action'] is action:
            data.append(d)
            observations.remove(d)
    ac1 = bestResponse[ch1]['accuracy']
    ac2 = bestResponse[ch2]['accuracy']
    MI = matMI[i][j] 
    #sorted_nextGeneration.remove(ch1)
    #sorted_nextGeneration.remove(ch2)

    knowledge = {'chromosomes':[ch1, ch2], 'action': action, 'data': data, 'accuracies':[ac1, ac2], 'MI':MI}
    CONTINUE = True
    return CONTINUE, knowledge, observations, bestResponse, sorted_nextGeneration

    '''  
    masks = []
    for i in xrange(len(nextGeneration)):
        mask = findMask(matMI, i) # ILS 
        masks.append(mask)
        print "\nRelated chromosomes:"
        for k in mask:
            chromosome = nextGeneration[k]
            printResponse(chromosome, bestResponse, features)
    ''' 

def findFeatures(observations): # observations = [ {'gameState': gameState, 'action': action}, ... ]
    from featureGenerator import generateFeatures
    features = generateFeatures()
    population = generateInitialPopulation(features, 100)
    knowledgeBase = [] #[{'chromosomes':[ch1, ch2], 'action': action, 'data': data}, {}]
    
    while(True):
        CONTINUE, knowledge, observations, bestResponse, nextGeneration = oneRun(observations, features, population)
        if not CONTINUE: break
        knowledgeBase.append(knowledge)
        print '\nKB:'
        for k in knowledgeBase:
            printKnowledge(k, features)
        #raw_input("\nPress <Enter> to continue...\n")
        #population = nextGeneration
        population = generateInitialPopulation(features, 100)

    with open('data.txt', 'wb') as outfile:
        pickle.dump(knowledgeBase, outfile)

def run():
    knowledgeBase = pickle.load( open('data.txt', 'rb') )
    print '\nKB:'
    for k in knowledgeBase:
        printKnowledge(k, features)
    print 'Running interviewer.py'
    l1 = [0,1,1,1]
    l2 = [0,1,1,1]
    from sklearn.metrics import normalized_mutual_info_score
    print normalized_mutual_info_score(l1, l2)

if __name__ == '__main__':
    run()
