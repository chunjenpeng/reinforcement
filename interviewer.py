import json, random
from operator import itemgetter
from itertools import product

import featureGenerator as fg
import numpy as np
from numpy import unravel_index

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


def generateInitialPopulation(features, n): #? String or List of float?
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
    
    satisfyStateList = satisfyState[bestAction] 
    bestData = {'bestAction':bestAction, 'accuracy':accuracy, 'satisfyStateList':satisfyStateList}
    return bestData


def steepestDescent(chromosome, features, observations, bestResponse):
    # bestResponse = { chromosome: {'bestAction':bestAction, 'accuracy':accuracy, 'satisfyStateList':satisfyStateList},  ...}
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


def MinimumDescriptionLength(population, features, observations, bestResponse):
    nextGeneration= [] 
    print '\nSteepest Descent for each chromosome...'
    for chromosome in population:
        child = steepestDescent(chromosome, features, observations, bestResponse) 
        print chromosome,
        while(child != chromosome):
            print '->', child,
            chromosome = child
            child = steepestDescent(chromosome, features, observations, bestResponse) 
        print '' 
        
        if child not in nextGeneration: 
            nextGeneration.append(child)
    
    return nextGeneration

def printPopulationResponses(population, bestResponse, features, topN):
    # bestResponse = { chromosome: {'bestAction':bestAction, 'accuracy':accuracy, 'satisfyStateList':satisfyStateList},  ...}
    responseWithAccuracy = [] 
    for chromosome in population:
        data = bestResponse[chromosome]
        response = (chromosome, data['bestAction']) 
        accuracy = data['accuracy']
        responseWithAccuracy.append((response, accuracy))
    
    responseWithAccuracy.sort(key=itemgetter(1), reverse=True)
    for i in xrange(topN):
        (chromosome, action), accuracy = responseWithAccuracy[i]
        print ('%8.7f' % accuracy), ':', chromosome, 'Go', action, ', when', chromosome2feature(chromosome, features)

def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log2(c_normalized))
    return H

def calc_MI(X,Y,bins):
    c_XY = np.histogram2d(X,Y,bins)[0]
    c_X = np.histogram(X,bins)[0]
    c_Y = np.histogram(Y,bins)[0]

    H_X = shan_entropy(c_X)
    H_Y = shan_entropy(c_Y)
    H_XY = shan_entropy(c_XY)

    MI = H_X + H_Y - H_XY
    return MI

def generateMatrix(A):
    n = A.shape[1]
    bins = n  
    matMI = np.zeros((n,n))
    for ix in np.arange(n):
        for jx in np.arange(ix+1, n):
            matMI[ix, jx] = calc_MI(A[:, ix], A[:,jx], bins)
            matMI[jx, ix] = calc_MI(A[:, ix], A[:,jx], bins)
    return matMI

def findMutualInformation(population, bestResponse):
    matrix = []
    for chromosome in population:
        satisfyStateList = bestResponse[chromosome]['satisfyStateList'] 
        matrix.append(satisfyStateList)
    arr = np.array(matrix)
    print '\nMutual Information Matrix:'
    matMI = generateMatrix(np.transpose(arr))
    np.set_printoptions(suppress=True, precision=3)
    #print matMI
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

def findFeatures(observations): # observations = [ {'gameState': gameState, 'action': action}, ... ]
    print '\nNumber of gameStates recorded:', len(observations)
    
    from featureGenerator import generateFeatures
    features = generateFeatures()
    population = generateInitialPopulation(features, 100)
    
    bestResponse = {} # { chromosome: {'bestAction':bestAction, 'accuracy':accuracy, 'satisfyStateList':satisfyStateList},  ...}
    nextGeneration = MinimumDescriptionLength(population, features, observations, bestResponse)
    
    print '\nLearned Features:'
    printPopulationResponses(nextGeneration, bestResponse, features, topN = len(nextGeneration))
    
    
    matMI = findMutualInformation(nextGeneration, bestResponse)
    relatedFeatureList = findRelatedFeatures(matMI)         # Related Feature List
    print 'Related features:' 
    for feature1, feature2 in relatedFeatureList:
        print '(',feature1,',',feature2,') = ', matMI[feature1][feature2], nextGeneration[feature1], nextGeneration[feature2]
    
    masks = []                                              # Masks
    for i in xrange(len(nextGeneration)):
        mask = findMask(matMI, i) # ILS 
        masks.append(mask)
    #    print mask
    print 'masks :', masks 
    

def run():
    print 'Running interviewer.py'

if __name__ == '__main__':
    findFeatures([])
