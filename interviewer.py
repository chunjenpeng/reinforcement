#author Daniel & Ryan

from pacman import GameState
from game import GameStateData
from game import Game
from game import Directions
from game import Actions

mazeHeight = 1
mazeLength = 5
posPacman = 1
posGhost = 4
listFood = []
listCapsule = []
gameData = dict(mazeLength = mazeLength, posPacman = posPacman, 
            posGhost = posGhost, listFood = listFood, listCapsule = listCapsule)

usedGameStates = []

def generateGameData(seed = 0):
    gameData = [mazeLength, posPacman, posGhost, listFood, listCapsule]
    return gameData
    
def ghostRule(gameData):
    return True

def generateLayout():
    layoutText = [None]*(2+mazeHeight)

    wall = ""

    for k in range(0,(mazeLength+2)):
        wall += "%"

    layoutText[0] = wall

    layoutText[mazeHeight+1] = wall

    print layoutText

def generateGameState(gameData):
    gamestate = new GameState()

    for k in range(1, mazeLength-1):

        gamestate = new GameState(self.generateLayout())

        if gamestate not in usedGameStates:

            usedGameStates.append(gamestate)
    
def generateGameStates():
    randomSeed = 0
    for repeat in range (0, 10, 1):
        gameData = generateGameData(randomSeed)
        if ghostRule(gameData):
            gameState = generateGameState(gameData)
            if gameState not in usedGameStates.keys():
                print gameState

print generateLayout()