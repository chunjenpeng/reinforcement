#author Daniel & Ryan

from pacman import GameState
from game import GameStateData
from game import Game
from game import Directions
from game import Actions

mazeLength = 5
posPacman = 1
posGhost = 4
listFood = []
listCapsule = []
gameData = [mazeLength, posPacman, posGhost, listFood, listCapsule]

usedGameStates = []

def generateGameData(seed = 0):
    gameData = [mazeLength, posPacman, posGhost, listFood, listCapsule]
    return gameData
    
def ghostRule(gameData):
    return True

def generateGameState(gameData):
    gamestate 
    if gamestate not in usedGameStates:
        return gamestate
    
def generateGameStates():
    randomSeed = 0
    for repeat in range (0, 10, 1):
        gameData = generateGameData(randomSeed)
        if ghostRule(gameData):
            gameState = generateGameState(gameData)
            if gameState not in usedGameStates.keys():
                print gameState
            