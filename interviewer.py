import json
import featureGenerator as fg
from operator import itemgetter
from itertools import product

def matchStatesWithChromosomes( population, features, observations):
    chromosomesWithData = {}
    for chromosome in population:
        chromosomesWithData[chromosome] = []
        for data in observations:    # ??efficiency??
            if fg.satisfyFeatures( chromosome, features, data['gameState'] ):
                chromosomesWithData[chromosome].append(data)
    return chromosomesWithData

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

def calculateAccuracy(chromosomesWithData): #Accuracy of chromosome or Accuracy of gameStates?
    #accuracyOfGameStates= {} 
    accuracyOfChromosomes = {} 
    for chromosome in chromosomesWithData.keys():
        aoc = {'North':0.0, 'South':0.0, 'East':0.0, 'West':0.0}
        for data in chromosomesWithData[chromosome]:
            action = data['action']
            aoc[action] += 1.0

        total_num = len(chromosomesWithData)
        for action in aoc:
            aoc[action] /= total_num
        accuracyOfChromosomes [chromosome] = aoc
    
    return accuracyOfChromosomes 

def sortByAction(accuracyOfChromosomes):
    chromosomesForActions = {'North':[], 'South':[], 'East':[], 'West':[]} 
    for chromosome in accuracyOfChromosomes:
        aoc = accuracyOfChromosomes[chromosome]
        for action in aoc:
            chromosomesForActions[action].append((chromosome, aoc[action]))
    for action in chromosomesForActions: 
        chromosomesForActions[action].sort(key=itemgetter(1)) # sorted by accuracy in [(ch, acc), (ch, acc), ...]
    return chromosomesForActions 
    
def findFeatures(observations):
    # observations = [ {'gameState': gameState, 'action': action}, ... ]
    from featureGenerator import generateFeatures
    features = generateFeatures()
    print 'findFeatures'
    population = generateInitialPopulation(features)
    chromosomesWithData = matchStatesWithChromosomes(population, features, observations)
    accuracyOfChromosomes = calculateAccuracy(chromosomesWithData)
    chromosomesForActions = sortByAction(accuracyOfChromosomes)
    for action in chromosomesForActions:
        print action, chromosomesForActions[action]


def run():
    print 'Running interviewer.py'

if __name__ == '__main__':
    run()
