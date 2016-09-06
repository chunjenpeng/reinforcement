import json
import featureGenerator as fg
from operator import itemgetter
from itertools import product

def chromosome2feature(chromosome, features):
    feature_string = ''  
    for i in xrange(len(chromosome)):
        if i > 0:
            feature_string += ' and '
        if chromosome[i] == '0':
            feature_string += 'Not '
        else:
            feature_string += '    '

        feature_string += features[i].__str__()
    return feature_string

def generateChromosomes(string):
    population = []
    c = string.count('*')
    string = string.replace('*', '{}')
    for x in product('01', repeat=c):
        chromosome = string.format(*x)
        population.append(chromosome)
    return population

def generateInitialPopulation(features): #? String or List of float?
    initialString = '*' * len(features) 
    population = generateChromosomes(initialString) 
    return population

def matchStatesWithChromosomes( population, features, observations):
    chromosomesWithData = {}
    for chromosome in population:
        chromosomesWithData[chromosome] = []
        for data in observations:    # ??efficiency??
            if fg.satisfyFeatures( chromosome, features, data['gameState'] ):
                chromosomesWithData[chromosome].append(data)
    return chromosomesWithData

def calculateAccuracy(chromosomesWithData): #Accuracy of chromosome or Accuracy of gameStates?
    #accuracyOfGameStates= {}
    accuracyOfChromosomes = {} 
    for chromosome in chromosomesWithData.keys():
        aoc = {'North':0.0, 'South':0.0, 'East':0.0, 'West':0.0, 'Stop':0.0}
        for data in chromosomesWithData[chromosome]:
            action = data['action']
            aoc[action] += 1.0

        data_num = len(chromosomesWithData[chromosome])
        print chromosome, ', data_num:', data_num
        if data_num != 0:
            for action in aoc:
                aoc[action] /= data_num
        accuracyOfChromosomes [chromosome] = aoc
    
    return accuracyOfChromosomes 

def sortByAction(accuracyOfChromosomes):
    chromosomesForActions = {'North':[], 'South':[], 'East':[], 'West':[], 'Stop':[]} 
    for chromosome in accuracyOfChromosomes:
        aoc = accuracyOfChromosomes[chromosome]
        for action in aoc:
            chromosomesForActions[action].append((chromosome, aoc[action]))
    for action in chromosomesForActions: 
        chromosomesForActions[action].sort(key=itemgetter(1), reverse=True) # sorted by accuracy in [(ch, acc), (ch, acc), ...]
    return chromosomesForActions 

def sortByAccuracy(chromosomesForActions, features):
    responses = []
    for action in chromosomesForActions:
        print action 
        for chromosome, accuracy in chromosomesForActions[action]:
            print ('%4.3f' % accuracy), chromosome
            response = (action, chromosome) 
            responses.append((response, accuracy))
    
    responses.sort(key=itemgetter(1), reverse=True)
    for response, accuracy in responses:
        print ('%4.3f' % accuracy), ': Go', response[0], 'when', chromosome2feature(response[1], features)
    return responses 

def findFeatures(observations): # observations = [ {'gameState': gameState, 'action': action}, ... ]
    from featureGenerator import generateFeatures
    features = generateFeatures()
    population = generateInitialPopulation(features)
    
    chromosomesWithData = matchStatesWithChromosomes(population, features, observations)
    accuracyOfChromosomes = calculateAccuracy(chromosomesWithData)
    chromosomesForActions = sortByAction(accuracyOfChromosomes)
    responses = sortByAccuracy(chromosomesForActions, features)

def run():
    print 'Running interviewer.py'

if __name__ == '__main__':
    run()
