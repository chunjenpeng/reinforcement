import json, random, sys
from operator import itemgetter
from itertools import product
import cPickle as pickle
import numpy as np
from numpy import unravel_index
from sklearn.metrics import normalized_mutual_info_score

import featureGenerator as fg
from pacman import GameState
'''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def plot3D(d):
    z,x,y = d.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, -z, zdir='z', c= 'red')
    plt.savefig("demo.png")
'''

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


def printResponse(response, features):
    '''
    response = {
        'chormosome':chromosome,
        'action': action,
        'totalDataNum': totalDataNum,
        'dataNum': dataNum,
        'accuracy':accuracy,
        'matchList':matchList
    }
    '''
    chromosome = response['chromosome']
    action = response['action']
    accuracy = response['accuracy']
    dataNum = response['dataNum'] 
    totalDataNum = response['totalDataNum']
    print ('%8.7f (%4d/%4d)' % (accuracy, dataNum, totalDataNum)), ':', chromosome
    print 'Go', action, ', when', chromosome2feature(chromosome, features)


def generateDescendants(string):
    descendants = []
    for i in xrange(len(string)):
        if string[i] != '*':
            child = list(string)
            child[i] = '*'
            descendants.append(''.join(child))
    return descendants


def generatePossibleChromosomes(string):
    population = []
    c = string.count('*')
    string = string.replace('*', '{}')
    for x in product('01', repeat=c):
        chromosome = string.format(*x)
        population.append(chromosome)
    return population


def generateInitialPopulation(features, n): 
    initialString = '*' * len(features) 
    population = generatePossibleChromosomes(initialString) 
    if n >= len(population):
        return population
    random.shuffle(population)
    return population[:n]


def generateTable(features, observations, actions):
    f = len(features)
    o = len(observations)
    a = len(actions)
    tableFeatureState = np.zeros((f,o))
    tableStateAction = np.zeros((o,a))
    table3D = np.zeros((a,o,f))
    for j in xrange(o):
        gameState = observations[j]['gameState']
        action = observations[j]['action']
        i = actions.index(action)
        #tableStateAction[j][k] = 1
        for k in xrange(f):
            if features[k].satisfy(gameState):
                #tableFeatureState[i][j] = 1
                table3D[i][j][k] = 1
            else:
                table3D[i][j][k] = -1
            print('generating table ... %d/%d\r'%(j*f+k,f*o)),
    #plot3D(table3D)
    return table3D, tableFeatureState, tableStateAction


def matchData(chromosome, table):
    ch = np.array(list(chromosome))
    location = np.where(ch != '*')[0]
    
    ch = []
    for c in chromosome:
        if c == '*':
            ch.append(0)
        elif c == '1':
            ch.append(1)
        elif c == '0':
            ch.append(-1) 
    ch = np.array(ch)
    #print 'table',table.shape
    A = table[:,:,location]
    #print 'A',A.shape
    B = ch[location].astype(int).T
    C = A*B
    np.place(C, C<0, 0)
    matchLists = np.prod(C, axis=2)
    dataNumList = np.sum(matchLists,axis=1)
    totalDataNum = np.sum(dataNumList)
    if location.size == 0: #empty list
        matchLists =  np.prod(table, axis=2)
        np.place(matchLists, matchLists<0, 1)
        dataNumList = np.sum(matchLists, axis=1)
        totalDataNum = np.sum(dataNumList)
    '''
    print 'chromosome\n', chromosome
    print 'location\n', location
    print 'features\n',B
    print 'extracted table\n',A
    print 'matchLists\n',matchLists
    print 'dataNumList\n',dataNumList
    print 'totalDataNum\n',totalDataNum
    '''
    return matchLists, dataNumList, totalDataNum

def generateResponses(chromosome, table, actions):
    responses = []
    matchLists, dataNumList, totalDataNum = matchData(chromosome, table) 
    for i in xrange(len(actions)):
        dataNum = dataNumList[i]
        accuracy = 0.0 if totalDataNum < 1 else dataNum/totalDataNum
        action = actions[i]
        response = {} 
        response['chromosome'] = chromosome
        response['action'] =  action
        response['dataNum'] =  dataNum
        response['totalDataNum'] =  totalDataNum
        response['accuracy'] = accuracy
        response['matchList'] = matchLists[i]
        responses.append(response)
        c = response
        #print c['chromosome'], c['accuracy'], '(', c['dataNum'], '/', c['totalDataNum'], ')'
    return responses

def generateBestResponse(chromosome, table, actions):
    responses = generateResponses(chromosome, table, actions)
    bestResponse = max(responses, key = lambda response: response['accuracy'])
    return bestResponse

def steepestDescent(chromosome, responseDict, gl):
    table = gl['table']
    actions = gl['actions']
    bestResponse = generateBestResponse(chromosome, table, actions)
    #responseDict[chromosome] = bestResponse 

    descendants = generateDescendants(chromosome)
    random.shuffle(descendants)
    for child in descendants:
        childResponse = generateBestResponse(child, table, actions)
        ''' 
        if child in bestResponse:
            childResponse = responseDict[child]
        else:
            childResponse = generateBestResponse(child, table)
            responseDict[child] = childResponse 
        ''' 
        #2016-10-20
        threshold = 0.5 #BEHOLD!!! WHY?0.5?
        if bestResponse['dataNum'] == 0 or \
           childResponse['accuracy'] >= bestResponse['accuracy'] or \
           childResponse['accuracy'] > threshold and childResponse['dataNum'] >= bestResponse['dataNum']:

            bestResponse = childResponse 
    
    #c = bestResponse
    #print 'BestResponse:', c['action']
    #print c['chromosome'], c['accuracy'], '(', c['dataNum'], '/', c['totalDataNum'], ')'
    #raw_input("")
    
    return bestResponse


def hillClimbing(gl, doPrint=False):
    population = gl['population']
    features = gl['features']
    # Record chromosome score
    responseDict = {} 
    winner = {} 
    if doPrint: print '\nSteepest Descent for each chromosome...'
    for chromosome in population:

        bestResponse = steepestDescent(chromosome, responseDict, gl) 
        if doPrint: print chromosome,
        while( bestResponse['chromosome'] != chromosome):
            chromosome = bestResponse['chromosome'] 
            if doPrint: print '->', chromosome,
            bestResponse = steepestDescent(chromosome, responseDict, gl) 
            #printResponse(bestResponse, features)
        if doPrint: print '' 
        
        if chromosome not in winner: 
            winner[chromosome] = bestResponse
    
    nextResponses = [ winner[c] for c in winner ] 
   
    return sorted(nextResponses, key = lambda response: (response['accuracy'],response['dataNum']), reverse=True)


def printKnowledge(knowledge, features):
    # knowledge = {'prior':str, 'responses': [response], 'dataNum': int, 'totalDataNum':int, 'accuracy':float}
    # response  = {'chormosome':str, 'action':str, 'totalDataNum':int, 'dataNum':int, 'accuracy':float, 'matchList':list }
    print '\nKnowledge:'
    print 'Given', chromosome2feature(knowledge['prior'], features)
    print 'accuracy:', knowledge['accuracy']

def generateKnowledge(gl, responses):
    prior = ''
    responses = []      #
    totalDataNum = 10   #data that fit prior
    dataNum = 1         #data that fit this knowledge
    accuracy = dataNum/totalDataNum
    knowledge = {'prior':prior, 'responses': responses, 'dataNum': dataNum, 'totalDataNum': totalDataNum, 'accuracy':accuracy}
    return knowledge

def oneRun(gl):
    observations = gl['observations']
    features = gl['features']

    print '\nNumber of gameStates:', len(observations)
    doPrint = True 
    
    nextResponses = hillClimbing(gl, doPrint)
    print '\nResponse Accuracy:'
    for response  in nextResponses:
        printResponse(response, features)
    
    knowledge = generateKnowledge(gl, nextResponses)
    #TODO: kick out observations
    return observations, nextResponses, knowledge
    
    ''' 
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


def findFeatures(observations): # observations = [ {'gameState': gameState, 'action': action}, ... ]
    filename = 'observations.p'
    with open(filename, 'wb') as outfile:
        pickle.dump(observations, outfile)
    
    actions = ['Stop', 'North', 'South', 'East', 'West']
    from featureGenerator import generateFeatures
    features = generateFeatures()
    population = generateInitialPopulation(features, 100)
    #population = ['11*0*']
    table3D, tableFeatureState, tableStateAction = generateTable(features, observations, actions)
    gl = {'features':features, 'population':population, 'observations':observations, 'actions': actions, 'table':table3D}

    responses = []
    knowledgeBase = []
    
    while(True):
        observations, nextResponses, knowledge = oneRun(gl)
        knowledgeBase.append(knowledge)
        for response in nextResponses:
            printResponse(response, features)
        gl['population'] = [ response['chromosome'] for response in nextResponses ]
        #population = generateInitialPopulation(features, 100)
        #print '\nKB:'
        #for k in knowledgeBase:
        printKnowledge(knowledge, features)
        raw_input("\nPress <Enter> to continue...\n")

    sorted_knowledgeBase = sorted(knowledgeBase, key=lambda k: (max(k['accuracies'][0], k['accuracies'][1]),len(k['data'])), reverse=True)
    filename = 'test.p'
    with open(filename, 'wb') as outfile:
        pickle.dump(sorted_knowledgeBase, outfile)
    

def run(filename):
    from featureGenerator import generateFeatures
    features = generateFeatures()
    
    savefile = open(filename, 'r') 
    knowledgeBase = pickle.load( savefile )
    for k in knowledgeBase:
        printKnowledge(k, features)

if __name__ == '__main__':
    if len(sys.argv) < 2 or len(sys.argv) > 2:
        print 'python interviewer.py <filename>'
    else:
        run(sys.argv[1])
'''
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
'''

