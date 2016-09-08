import json, random
import featureGenerator as fg
from operator import itemgetter
from itertools import product

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


def printArray(population, bestResponse):
    for chromosome in population:
        print chromosome, bestResponse[chromosome]['satisfyStateList']

def findFeatures(observations): # observations = [ {'gameState': gameState, 'action': action}, ... ]
    print '\nNumber of gameStates recorded:', len(observations)
    
    from featureGenerator import generateFeatures
    features = generateFeatures()
    population = generateInitialPopulation(features, 100)
    
    bestResponse = {} # { chromosome: {'bestAction':bestAction, 'accuracy':accuracy, 'satisfyStateList':satisfyStateList},  ...}
    nextGeneration = MinimumDescriptionLength(population, features, observations, bestResponse)
    
    print '\nLearned Features:'
    printPopulationResponses(nextGeneration, bestResponse, features, topN = len(nextGeneration))
    #printArray(nextGeneration, bestResponse)
    

def run():
    print 'Running interviewer.py'

if __name__ == '__main__':
    findFeatures([])
