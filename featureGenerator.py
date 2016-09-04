############# Remove this part when featureGenerator.py is finished #######################
class gameData:
    def __init__(self, args, chromosome = None):
        self.initialize(args)
        if chromosome != None:
            while satisfyFeatures(generateFeatures(), self, chromosome) == False:
                self.initialize(args)
    
    def initialize(self, args):
        self.mazeHeight = args['mazeHeight']
        self.mazeLength = args['mazeLength']
        if args['posPacman'] is None:
            self.posPacman = randint(1, self.mazeLength)
        else:
            self.posPacman = args['posPacman']
        if args['posGhost'] is None:
            self.posGhost = (((self.posPacman-1+randint(1, self.mazeLength-1)))%self.mazeLength) + 1
        else:
            self.posGhost = args['posGhost']
        self.listFood = []
        self.listCapsule = []
        for k in range(1,self.mazeLength+1): #randomization of food and capsules
            if k == self.posPacman or k == self.posGhost:
                continue
            #random.seed(0)
            randomInt = randint(0,2)
            if randomInt == 1:
               self.listCapsule.append(k)
            elif randomInt == 2:
               self.listFood.append(k)

    def initializeWithState(self, gameState):
        layout = gameState.data.layout.layoutText
        self.mazeHeight = len(layout)-2
        self.mazeLength = len(layout[0])-2
        self.listFood = []
        self.listCapsule = []

        pacman = 0
        for k in range(1, len(layout[1])):
            if layout[1][k] == "P":
                pacman = k
        self.posPacman = pacman

        ghost = 0
        for k in range(1, len(layout[1])):
            if layout[1][k] == "G":
                ghost = k
        self.posGhost = ghost

        for k in range(1, len(layout[1])):
            if layout[1][k] == ".":
                self.listFood.append(k)

        for k in range(1, len(layout[1])):
            if layout[1][k] == "o":
                self.listCapsule.append(k)
        
        return self



class Feature:
    def __init__(self, closure):
        self.func = closure
    def satisfy(self, gameData):
        return self.func(gameData)

def satisfyFeatures(chromosome, features, gameState):
    return True
    for feature in features:
        #TODO
        if features[feature].satisfy(gameData) is not chromosome[feature]:
            return False
    return True

def isNear(pos1, pos2, near):
    if (abs(pos1 - pos2)<= near):
        return True
    return False

def atEast(pos1, pos2):
    if (pos1 > pos2):
        return True
    return False
    
def atCorner(pos, length):
    if(pos == 1 or pos == length):
        return True
    return False

def pacmanAtCorner(gameData):
    return atCorner(gameData.posPacman, gameData.mazeLength)

def ghostIsNear(gameData):
    return isNear( gameData.posGhost, gameData.posPacman, near = 1 )

def ghostAtEast(gameData):
    return atEast(gameData.posGhost, gameData.posPacman)

def ghostAtCorner(gameData):
    return atCorner(gameData.posGhost, gameData.mazeLength)

def closestFoodIsNear(gameData, near = 1):
    return util.closest(gameData.listFood,args['mazeLength'],gameData) <= near

def closestFoodAtEast(gameData):
    closestList = util.closestList(gameData.listFood,args['mazeLength'],gameData)
    if (len(closestList) == 0):
        return False
    if (len(closestList) == 1):
        if (gameData.posPacman < closestList[0]):
            return True
        return False
    if (len(closestList) == 2):
        return True
    
def closestCapsuleIsNear(gameData, near = 1):
    return util.closest(gameData.listCapsule, args['mazeLength'], gameData) == near

def closestCapsuleAtEast(gameData):
    closestList = util.closestList(gameData.listCapsule,args['mazeLength'],gameData)
    if(len(closestList)==0):
        return False
    if(len(closestList)==1):
        if(gameData.posPacman < closestList[0]):
            return True
        return False
    if(len(closestList)==2):
        return True

def closestWallIsNear(gameData, near = 1):
    return atCorner(gameData.posPacman, gameData.mazeLength)

def closestWallAtEast(gameData):
    if(gameData.posPacman >= gameData.mazeLength/2):
        return True
    else:
        return False

def generateFeatures():
    features = {}
    features['ghostIsNear'] = Feature(ghostIsNear)
    features['ghostAtEast'] = Feature(ghostAtEast)
    features['closestFoodIsNear'] = Feature(closestFoodIsNear)
    features['closestFoodAtEast'] = Feature(closestFoodAtEast)
    features['closestCapsuleIsNear'] = Feature(closestCapsuleIsNear)
    features['closestCapsuleAtEast'] = Feature(closestCapsuleAtEast)
    features['closestWallIsNear'] = Feature(closestWallIsNear)
    features['closestWallAtEast'] = Feature(closestWallAtEast)
    #features['pacmanAtCorner'] = Feature(pacmanAtCorner) 
    return features
