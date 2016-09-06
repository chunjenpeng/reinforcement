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
    for data in observations:    # ??efficiency??
        if fg.satisfyFeatures( chromosome, features, data['gameState'] ):
            chromosomeData.append(data)
    return chromosomeData

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
    chromosomeData = matchDataWithChromosome(chromosome, features, observations)
    allActionAccuracy = calculateActionAccuracy(chromosomeData)
    bestAction = max(allActionAccuracy, key=allActionAccuracy.get) 
    accuracy = allActionAccuracy[bestAction]
    return (bestAction,accuracy)


def steepestDescent(chromosome, features, observations, chromosomeBestAction):
    bestChromosome = chromosome
    bestAction, accuracy = findBestAction(chromosome, features, observations) 
    chromosomeBestAction[chromosome] = (bestAction, accuracy)
    
    descendants = generateDescendants(chromosome)
    for child in descendants:
        if child in chromosomeBestAction:
            action, acc = chromosomeBestAction[child]
        else:
            action, acc = findBestAction(child, features, observations) 
            chromosomeBestAction[child] = (action, acc)
        
        if acc >= accuracy:
            bestChromosome, accuracy = child, acc 
            bestAction = action
    
    return bestChromosome, chromosomeBestAction


def MinimumDescriptionLength(population, features, observations):
    chromosomeBestAction = {}
    bestChromosomes = [] 
    responseWithAccuracy = [] 
    for chromosome in population:
        #Steepest Descent
        child, chromosomeBestAction = steepestDescent(chromosome, features, observations, chromosomeBestAction) 
        print chromosome,
        while(child != chromosome):
            print '->', child,
            chromosome = child
            child, chromosomeBestAction = steepestDescent(chromosome, features, observations, chromosomeBestAction) 
        print '' 
        
        if child not in bestChromosomes: 
            bestChromosomes.append(child)
            bestAction, accuracy = chromosomeBestAction[child]
            response = (child, bestAction) 
            responseWithAccuracy.append((response, accuracy))
    
    return responseWithAccuracy

def printResponses(responseWithAccuracy, features, topN):
    for i in xrange(topN):
        (chromosome, action), accuracy = responseWithAccuracy[i]
        print ('%8.7f' % accuracy), ':', chromosome, 'Go', action, ', when', chromosome2feature(chromosome, features)

def findFeatures(observations): # observations = [ {'gameState': gameState, 'action': action}, ... ]
    print 'number of data:', len(observations)
    
    from featureGenerator import generateFeatures
    features = generateFeatures()
    population = generateInitialPopulation(features, 100)
    responseWithAccuracy = MinimumDescriptionLength(population, features, observations)
    
    print '\nLearned Features:'
    responseWithAccuracy.sort(key=itemgetter(1), reverse=True)
    printResponses(responseWithAccuracy, features, topN = len(responseWithAccuracy))
    

def run():
    print 'Running interviewer.py'

if __name__ == '__main__':
    findFeatures([])
