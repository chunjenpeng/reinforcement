# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """

  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    currentPos = currentGameState.getPacmanPosition()
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()    
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
   

    "*** YOUR CODE HERE ***"
    score = ghostDistance = oldFoodDistance = newFoodDistance = capsuleDistance = 0
    #evaluate ghost distance
    newGhostPositions = successorGameState.getGhostPositions()      
    for newGhostPosition in newGhostPositions :
      ghostDistance += manhattanDistance( newPos, newGhostPosition )
      if (manhattanDistance(newPos, newGhostPosition) == 1):
        score -= 100
    
    #score -= 1/ghostDistance/3
        
    #evaluate food distance 
    oldFoodNum = currentGameState.getNumFood()
    newFoodNum = successorGameState.getNumFood()
    if(oldFoodNum == newFoodNum):
      score -= 20
    newFood = successorGameState.getFood()
    newFoodPositions = []
    
    for x in range(newFood.width):
      for y in range(newFood.height):
        if(newFood[x][y] == True):
          newFoodPositions.append((x,y))
    
    oldNearestFood = newNearestFood = 10000000    
    for newFoodPosition in newFoodPositions :
      if(manhattanDistance(currentPos, newFoodPosition) < oldNearestFood):
        oldNearestFood = manhattanDistance(currentPos, newFoodPosition)
      if(manhattanDistance(newPos, newFoodPosition) < newNearestFood):
        newNearestFood = manhattanDistance(newPos, newFoodPosition)
        
      oldFoodDistance += manhattanDistance(currentPos, newFoodPosition)
      newFoodDistance += manhattanDistance(newPos, newFoodPosition)
      score += 1/manhattanDistance(newPos, newFoodPosition)
    
    #try find the nearest food
    if(newNearestFood < oldNearestFood):
       score += 20
    #try not to wander around
    if(newFoodDistance >= oldFoodDistance):
      score -= 10
    
    #check not to be stuck in the middle
    currentMobility = len(currentGameState.getLegalActions(0)) #mobility
    successorMobility = len(successorGameState.getLegalActions(0))
    if(successorMobility == 1):
      score -= 50
    
#    print "successorGameState :\n{0}" .format(successorGameState)
#    print "newPos :\n{0}" .format(newPos)
#    print "newGhostPositions :\n{0}" .format(newGhostPositions)
#    print "ghostDistance :\n{0}" .format(ghostDistance)
#    print "score :\n{0}" .format(score)
#    print "getScore() :\n {0}" .format(successorGameState.getScore()+score)
#    print ""
#    print "==========================="   
    
    return successorGameState.getScore() + score
    

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    minmax = self.minimax(gameState, self.depth, 0)
#    print "minimax({0},{1})" .format(minmax[0], minmax[1])
#    print "GameState :\n{0}\n\n" .format(gameState)
    return minmax[1]


  def minimax(self, gameState, depth, agentIndex):
#    print "+++++++++++++++++++++++++++++++++++++"
#    print "GameState :\n{0}" .format(gameState)
#    print "depth : {0}" .format(depth)
#    print "agentIndex : {0}" .format(agentIndex)
#    print "-------------------------------------"
    
    if (depth == 1 and agentIndex == gameState.getNumAgents()-1) or gameState.isLose() or gameState.isWin() :
#      print "`````````````````````````````````````"
#      print "depth : {0}" .format(depth)
#      print "agentIndex : {0}" .format(agentIndex)
#      print "return [{0}, {1}]" .format(self.evaluationFunction(gameState), 0)   
#      print "`````````````````````````````````````"
      return (self.evaluationFunction(gameState), 0)
    if (agentIndex == 0):
      bestAction = Directions.STOP
      bestValue = -10000000000
      legalActions = gameState.getLegalActions(agentIndex)
      if Directions.STOP in legalActions: legalActions.remove(Directions.STOP)
#      for action in legalActions: print "action = {0}" .format(action)
      for action in legalActions:
#        if(action == Directions.STOP):
#          continue
#        else:
          successorGameState = gameState.generateSuccessor(agentIndex, action)
          val = self.minimax(successorGameState, depth, agentIndex+1)[0]
          if(val > bestValue):
            bestAction = action
            bestValue = val           
#      print "+++++++++++++++++++++++++++++++++++++"
#      print "GameState :\n{0}" .format(gameState)
#      print "depth : {0}" .format(depth)
#      print "agentIndex : {0}" .format(agentIndex)
#      print "-------------------------------------"
#      print "====================================="
#      print "depth : {0}" .format(depth)
#      print "agentIndex : {0}" .format(agentIndex)
#      print "return ({0}, {1})" .format(bestValue, bestAction)    
#      print "====================================="
      return (bestValue, bestAction)
          
    else:
      bestAction = Directions.STOP
      bestValue = val = 10000000000
      legalActions = gameState.getLegalActions(agentIndex)
#      for action in legalActions: print "action = {0}" .format(action)
      if Directions.STOP in legalActions: legalActions.remove(Directions.STOP)
      for action in legalActions:
 #       if(action == Directions.STOP):
 #         continue
 #       else:
 #         print "action = {0}" .format(action)
          successorGameState = gameState.generateSuccessor(agentIndex, action)
          if (agentIndex == gameState.getNumAgents()-1):
            val = self.minimax(successorGameState, depth-1, 0)[0]
#            print "depth : {0}" .format(depth)
#            print "agentIndex : {0}" .format(agentIndex)
#            print "val : {0}" .format(val)
#            print "bestValue : {0}" .format(bestValue)
          else:
            val = self.minimax(successorGameState, depth, agentIndex+1)[0]

#          print "depth : {0}" .format(depth)
#          print "agentIndex : {0}" .format(agentIndex)
#          print "val : {0}" .format(val)
#          print "bestValue : {0}" .format(bestValue)
          if(val < bestValue):
            bestAction = action
            bestValue = val
#      print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
#      print "depth : {0}" .format(depth)
#      print "agentIndex : {0}" .format(agentIndex)
#      print "return ({0}, {1})" .format(bestValue, bestAction)   
#      print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
            
      return (bestValue, bestAction)

          
        
class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    a = -10000000000
    b = 10000000000
    AlphaBeta = self.alphabeta(gameState, self.depth, 0, a, b )
    print "AlphaBeta({0},{1})" .format(AlphaBeta[0], AlphaBeta[1])
#    print "GameState :\n{0}\n\n" .format(gameState)
    return AlphaBeta[1]
  
  def alphabeta(self, gameState, depth, agentIndex, alpha, beta):
    if (depth == 1 and agentIndex == gameState.getNumAgents()-1) or gameState.isLose() or gameState.isWin() :
      return (self.evaluationFunction(gameState), 0)

    if (agentIndex == 0):
      bestAction = Directions.STOP
      val = -10000000000
      legalActions = gameState.getLegalActions(agentIndex)
      if Directions.STOP in legalActions: legalActions.remove(Directions.STOP)
      for action in legalActions:
        successorGameState = gameState.generateSuccessor(agentIndex, action)
        new_val = self.alphabeta(successorGameState, depth, agentIndex+1, alpha, beta )[0]
        
        #update val and bestAction
        if(new_val > val):
          bestAction = action
          val = new_val

        #pruning!!
        if val > alpha : 
          alpha = val
        if beta <= alpha:
          break #beta cut-off
          
      return (val, bestAction)
          
    else:
      bestAction = Directions.STOP
      val = 10000000000
      legalActions = gameState.getLegalActions(agentIndex)
      if Directions.STOP in legalActions: legalActions.remove(Directions.STOP)
      for action in legalActions:
        successorGameState = gameState.generateSuccessor(agentIndex, action)
        if (agentIndex == gameState.getNumAgents()-1):
          new_val = self.alphabeta(successorGameState, depth-1, 0, alpha, beta )[0]
        else:
          new_val = self.alphabeta(successorGameState, depth, agentIndex+1, alpha, beta )[0]

        #update val and bestAction
        if(new_val < val):
          bestAction = action
          val = new_val
          
        #pruning!!
        if val < beta : 
          beta = val
        if beta <= alpha:
          break #alpha cut-off
            
      return (val, bestAction)
  

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
#    util.raiseNotDefined()
    expectimax = self.expectimax(gameState, self.depth, 0)
#    print "minimax({0},{1})" .format(expectimax[0], expectimax[1])
#    print "GameState :\n{0}\n\n" .format(gameState)
    return expectimax[1]
    
    
    
  def expectimax(self, gameState, depth, agentIndex):
    if (depth == 1 and agentIndex == gameState.getNumAgents()-1) or gameState.isLose() or gameState.isWin() :
      return (self.evaluationFunction(gameState), 0)

    if (agentIndex == 0):
      bestAction = Directions.STOP
      bestValue = -10000000000
      legalActions = gameState.getLegalActions(agentIndex)
      if Directions.STOP in legalActions: legalActions.remove(Directions.STOP)
      for action in legalActions:
        successorGameState = gameState.generateSuccessor(agentIndex, action)
        val = self.expectimax(successorGameState, depth, agentIndex+1)[0]
        if(val > bestValue):
          bestAction = action
          bestValue = val           
      return (bestValue, bestAction)
          
    else:
#      bestAction = Directions.STOP
      meanValue = 0
      legalActions = gameState.getLegalActions(agentIndex)
      if Directions.STOP in legalActions: legalActions.remove(Directions.STOP)
      for action in legalActions:
        successorGameState = gameState.generateSuccessor(agentIndex, action)
        if (agentIndex == gameState.getNumAgents()-1):
          meanValue += self.expectimax(successorGameState, depth-1, 0)[0]
        else:
          meanValue += self.expectimax(successorGameState, depth, agentIndex+1)[0]
#          if(val < bestValue):
#            bestAction = action
#            bestValue = val
      meanValue = meanValue / len(legalActions)
      return (meanValue, 0)



def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    
    1.Pacman status evalutation
        legalActionsNum : mobility, better to have more choices of actions 
    
    2.FOOD status evaluation
        foodPositions :
        foodDistance  : better to have food near, 
                        poportionate to 1/(Distance^2)
        nearestFoodDistance
        A*?
    
    3.GHOST status evaluation
        ghostsDistance   :
        ghostPositions   :
        ghostsDistance   :
        capsulePositions :
        scaredTimes      :
  """
  "*** YOUR CODE HERE ***" 
  #1.Pacman status evalutation
  Score = 0    
  currentPos = currentGameState.getPacmanPosition()
  legalActionsNum = len(currentGameState.getLegalActions(0)) #mobility


    
  #2.FOOD status evaluation
  foodDistance = nearestFoodDistance = 0
  foodNum = currentGameState.getNumFood()
  foodGrid = currentGameState.getFood()
  nearestFood, foodPositions = (0,0), []  
  for x in range(foodGrid.width):
    for y in range(foodGrid.height):
      if(foodGrid[x][y] == True):
        foodPositions.append((x,y))
        
  for foodPosition in foodPositions:
    d = manhattanDistance(currentPos, foodPosition)
    foodDistance += d
    if (nearestFoodDistance == 0) : 
      nearestFoodDistance = d
      nearestFood = foodPosition
    elif nearestFoodDistance < d :
      nearestFoodDistance = d
      nearestFood = foodPosition
  
  blocked_X = blocked_Y = 0
  if nearestFood[0] - currentPos[0] > 0:
    blocked_X = currentPos[0] + 1
  else:
    blocked_X = currentPos[0] - 1
  if nearestFood[1] - currentPos[1] > 0:
    blocked_Y = currentPos[1] + 1
  else:
    blocked_Y = currentPos[1] - 1
  

  if nearestFoodDistance < 6:  
    blocked = currentGameState.hasWall(blocked_X, blocked_Y)
    if blocked:
      #print "blocked!!!"
      Score -= 20
    #2-1 better to have food near, poportionate to 1/(Distance^2)
    #Score += FOODDISTANCE_K1/(d*d)
    #Score += FOODDISTANCE_K/(d)
  
  #2-2 total food distance
  #Score += FOODDISTANCE_K2/foodDistance
  
  #2-3 try to find nearest food
  #Score += NEARESTFOODDISTANCE_K / nearestFoodDistance     

    
  #3.GHOST status evaluation 
  nearestScaredGhostDistance = nearestActiveGhostDistance = float("inf")
  ghostsDistance = capsulesDistance = 0
  ghostPositions = currentGameState.getGhostPositions() 
  scaredGhosts, activeGhosts = [], []
  for ghost in currentGameState.getGhostStates():
    if not ghost.scaredTimer:
      activeGhosts.append(ghost)
    else: 
      scaredGhosts.append(ghost)  
  
  
  def getManhattanDistances(ghosts): 
    return map(lambda g: util.manhattanDistance(currentPos, g.getPosition()), ghosts)
  
  #sum of ghost Distance
  #ghostsDistance += getManhattanDistances(ghostPositions)
  
  #nearestActiveGhostDistance
  if not activeGhosts :
    nearestActiveGhostDistance = float("inf")
  else :
    nearestActiveGhostDistance = min(getManhattanDistances(activeGhosts))

  #  nearestScaredGhostDistance
  if not scaredGhosts :
    nearestScaredGhostDistance = 0
  else :
    nearestScaredGhostDistance = min(getManhattanDistances(scaredGhosts))
      
  capsulePositions = currentGameState.getCapsules()
  capsuleNum = len(capsulePositions)
  scaredTimes = [ghostState.scaredTimer for ghostState in currentGameState.getGhostStates()]  
  
   
  if (legalActionsNum == 1):
    L = -20
  else:
    L  = 0

  Score = 1                * currentGameState.getScore() + \
          L                * legalActionsNum + \
          -4               * foodNum + \
          -50              * capsuleNum + \
          -1.5             * nearestFoodDistance + \
          -0               * 1/(foodDistance*foodDistance+1) + \
          -2               * (1.0/(nearestActiveGhostDistance+0.0001)) + \
          -4               * nearestScaredGhostDistance
  """             
  print "currentGameState :\n{0}" .format(currentGameState)
  print "legalActionsNum :\n{0}" .format(legalActionsNum)
  print "foodNum :\n{0}" .format(foodNum)
  print "capsuleNum :\n{0}" .format(capsuleNum)
  print "nearestFoodDistance :\n{0}" .format(nearestFoodDistance)
  print "nearestActiveGhostDistance :\n{0}" .format(nearestActiveGhostDistance)
  print "nearestScaredGhostDistance :\n{0}" .format(nearestScaredGhostDistance)
  print "getScore() :\n {0}" .format(currentGameState.getScore()+Score)
  print "Score :\n{0}" .format(Score)
  print "==========================="   
  """  
  return Score

# Abbreviation
better = betterEvaluationFunction


import sys
sys.path.append("D:\NEAT")
from neat import nn
from neat import population, visualize
from neat.config import Config

class NNAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    legalActions = gameState.getLegalActions(0)
    if Directions.STOP in legalActions: legalActions.remove(Directions.STOP)
    
    inputs = []
    outputs = legalActions
    
    
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config = Config(os.path.join(local_dir, 'nn_config'))
    
    pop = population.Population(config)
    pop.epoch(fitness_function, 1000)
    
    def eval_fitness(genomes):
        for g in genomes:
            net = nn.create_feed_forward_phenotype(g)
            g.fitness = gamestate.score
         
            output = net.serial_activate(intputs)




















    
    
    "*** YOUR CODE HERE ***"
#    util.raiseNotDefined()
    expectimax = self.expectimax(gameState, self.depth, 0)
#    print "minimax({0},{1})" .format(expectimax[0], expectimax[1])
#    print "GameState :\n{0}\n\n" .format(gameState)
    return expectimax[1]
    
    
    
  def expectimax(self, gameState, depth, agentIndex):
    if (depth == 1 and agentIndex == gameState.getNumAgents()-1) or gameState.isLose() or gameState.isWin() :
      return (self.evaluationFunction(gameState), 0)

    if (agentIndex == 0):
      bestAction = Directions.STOP
      bestValue = -10000000000
      legalActions = gameState.getLegalActions(agentIndex)
      if Directions.STOP in legalActions: legalActions.remove(Directions.STOP)
      for action in legalActions:
        successorGameState = gameState.generateSuccessor(agentIndex, action)
        val = self.expectimax(successorGameState, depth, agentIndex+1)[0]
        if(val > bestValue):
          bestAction = action
          bestValue = val           
      return (bestValue, bestAction)
          
    else:
#      bestAction = Directions.STOP
      meanValue = 0
      legalActions = gameState.getLegalActions(agentIndex)
      if Directions.STOP in legalActions: legalActions.remove(Directions.STOP)
      for action in legalActions:
        successorGameState = gameState.generateSuccessor(agentIndex, action)
        if (agentIndex == gameState.getNumAgents()-1):
          meanValue += self.expectimax(successorGameState, depth-1, 0)[0]
        else:
          meanValue += self.expectimax(successorGameState, depth, agentIndex+1)[0]
#          if(val < bestValue):
#            bestAction = action
#            bestValue = val
      meanValue = meanValue / len(legalActions)
      return (meanValue, 0)


