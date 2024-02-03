# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        if successorGameState.isWin():
            return 9999999999999
        
        newGhostDistances = [(newPos[0]-ghost.configuration.pos[0])**2+(newPos[1]-ghost.configuration.pos[1])**2 for ghost in newGhostStates]
        newSumGhostDistances=  sum([-dist if scared_time>0 else dist for dist,scared_time in zip(newGhostDistances, newScaredTimes)])
        
        try:
            newMinGhostDistances = min([dist for dist, scared_time in zip(newGhostDistances, newScaredTimes) if scared_time==0])
        except ValueError:
            newMinGhostDistances = 0
            
        currPos = currentGameState.getPacmanPosition()
        currGhostStates = currentGameState.getGhostStates()
        currScaredTimes = [ghostState.scaredTimer for ghostState in currGhostStates]
           
        currGhostDistances = [(newPos[0]-ghost.configuration.pos[0])**2+(newPos[1]-ghost.configuration.pos[1])**2 for ghost in currGhostStates]
        currSumGhostDistances=  sum([-dist if scared_time>0 else dist for dist,scared_time in zip(currGhostDistances, currScaredTimes)])
        
        try:
            currMinGhostDistances = min([dist for dist,scared_time in zip(currGhostDistances, currScaredTimes) if scared_time==0])
        except ValueError:
            currMinGhostDistances = 0
        
        newFoodDistScore = [(newPos[0]-food[0])**2+(newPos[1]-food[1])**2 for food in newFood.asList()]
        newMinFoodDistances = min(newFoodDistScore)
        
        currFoodDistScore = [(currPos[0]-food[0])**2+(currPos[1]-food[1])**2 for food in newFood.asList()]
        currMinFoodDistances = min(currFoodDistScore)

        currScore = 0
        
        if len(successorGameState.getCapsules()) > 0:
            
            newCapsuleDistScore = [(newPos[0]-capsule[0])**2+(newPos[1]-capsule[1])**2 for capsule in successorGameState.getCapsules()]
            newMinCapsuleFoodDistances = min(newCapsuleDistScore)
            
            currCapsuleDistScore = [(currPos[0]-capsule[0])**2+(currPos[1]-capsule[1])**2 for capsule in currentGameState.getCapsules()]
            currMinCapsuleFoodDistances = min(currCapsuleDistScore)
        
            if newMinCapsuleFoodDistances < currMinCapsuleFoodDistances:
                currScore+=20
        
        if len(currentGameState.getCapsules()) > 0 and newPos in currentGameState.getCapsules():
            currScore += 50
            
        if newMinFoodDistances < currMinFoodDistances:
            currScore += 200 - newMinFoodDistances
        
        currScore += (successorGameState.getScore() - currentGameState.getScore())
        
        if newSumGhostDistances < currSumGhostDistances:
            currScore += 50
        if newMinGhostDistances < currMinGhostDistances:
            currScore += 150
        
            
        if len(newFood.asList()) < len(currentGameState.getFood().asList()):
            currScore += 200
        
        currScore -= len(newFood.asList()) * 10
    
        return currScore

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        
        
        def minValue(agentIndex, gameState, depth):
            """
            Function that would get a minimum action value for the current agentIndex
            """
            legalActions = gameState.getLegalActions(agentIndex)

            if len(legalActions) == 0:
                # There are no legal actions to be made, therefore we return the value for the current gameState
                return self.evaluationFunction(gameState)

            possibleMinValues = []
            
            for action in legalActions:
                if agentIndex == gameState.getNumAgents() - 1:
                    # If it is, make minimum move for current agent and search for maximum move for Pacman
                    possibleMinValues.append(maxValue(gameState.generateSuccessor(agentIndex, action), depth))
                else:
                    # Otherwise, make next agemt move 
                    possibleMinValues.append(minValue(agentIndex + 1, gameState.generateSuccessor(agentIndex, action), depth))


            return min(possibleMinValues)


        def maxValue(gameState, depth):
            """
            Function that would get a maximum action value for pacman
            """
            
            pacmanIndex = 0
            legalActions = gameState.getLegalActions(pacmanIndex)

            # If no legal actions are possible, or we have reached the maximum depth possible, return the evaluation for the current gameState
            if depth == self.depth or len(legalActions) == 0:
                return self.evaluationFunction(gameState)


            possibleMaximumValues = []
            
            # Iterate through all possible actions and calculate their values
            for action in legalActions:
                possibleMaximumValues.append(minValue(pacmanIndex + 1, gameState.generateSuccessor(pacmanIndex, action), depth + 1))

            # Return the action which would get the maximum value
            return max(possibleMaximumValues)


        # Get all legal actions for pacman
        actionsPacman = gameState.getLegalActions(0)
        
        allActionsPacman = {}
        
        # Iterate through all possible moves for pacman for the 1st agent and add to the depth 
        for action in actionsPacman:
            allActionsPacman[action] = minValue(1, gameState.generateSuccessor(0, action), 1)
            

        # Return the action that gets the best value
        return max(allActionsPacman, key=allActionsPacman.get)
    
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        
        def minValue(agentIndex, gameState, depth, alpha, beta):
            """
            Function that would get a minimum action value for the current agentIndex
            """
            legalActions = gameState.getLegalActions(agentIndex)

            if len(legalActions) == 0:
                # There are no legal actions to be made, therefore we return the value for the current gameState
                return self.evaluationFunction(gameState)

            possibleMinValues = []
            
            for action in legalActions:
                if agentIndex == gameState.getNumAgents() - 1:
                    # If it is, make minimum move for current agent and search for maximum move for Pacman
                    currValue = maxValue(gameState.generateSuccessor(agentIndex, action), depth, alpha, beta)
                    
                else:
                    # Otherwise, make next agemt move 
                    currValue = minValue(agentIndex + 1, gameState.generateSuccessor(agentIndex, action), depth, alpha, beta)
                
                # If the value of the current action is lower than alpha, return it
                if currValue < alpha:
                    return currValue
                
                # Update beta
                beta = min(currValue, beta)
                possibleMinValues.append(currValue)


            return min(possibleMinValues)


        def maxValue(gameState, depth, alpha, beta):
            """
            Function that would get a maximum action value for pacman
            """
            
            pacmanIndex = 0
            legalActions = gameState.getLegalActions(pacmanIndex)

            # If no legal actions are possible, or we have reached the maximum depth possible, return the evaluation for the current gameState
            if depth == self.depth or len(legalActions) == 0:
                return self.evaluationFunction(gameState)


            possibleMaximumValues = []
            
            # Iterate through all possible actions and calculate their values
            for action in legalActions:
                currValue = minValue(pacmanIndex + 1, gameState.generateSuccessor(pacmanIndex, action), depth + 1, alpha, beta)
                
                # If the value of the current action is higher than beta, return it
                if currValue > beta:
                    return currValue
                
                # Update alpha
                alpha = max(currValue, alpha)
                
                possibleMaximumValues.append(currValue)
            
            return max(possibleMaximumValues)


        # Get all legal actions for pacman
        actionsPacman = gameState.getLegalActions(0)
        
        allActionsPacman = {}
        
        # Initialize alpha and beta with arbitrary numbers
        alpha = -9999999
        beta = 9999999
        
        
        # Iterate through all possible moves for pacman for the 1st agent and add to the depth 
        for action in actionsPacman:
            allActionsPacman[action] = minValue(1, gameState.generateSuccessor(0, action), 1, alpha, beta)
            
            # If the value of the current action is higher than beta, return it
            if allActionsPacman[action] > beta:
                return action
            
            # Update alpha
            alpha = max(alpha, allActionsPacman[action])

        # Return the action that gets the best value
        return max(allActionsPacman, key=allActionsPacman.get)
    
    
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
          
        
        def expValue(agentIndex, gameState, depth):
            """
            Function that would get a minimum action value for the current agentIndex
            """
            legalActions = gameState.getLegalActions(agentIndex)

            if len(legalActions) == 0:
                # There are no legal actions to be made, therefore we return the value for the current gameState
                return self.evaluationFunction(gameState)

            possibleMinValues = []
            
            for action in legalActions:
                if agentIndex == gameState.getNumAgents() - 1:
                    # If it is, make minimum move for current agent and search for maximum move for Pacman
                    possibleMinValues.append(maxValue(gameState.generateSuccessor(agentIndex, action), depth))
                else:
                    # Otherwise, make next agemt move 
                    possibleMinValues.append(expValue(agentIndex + 1, gameState.generateSuccessor(agentIndex, action), depth))


            return sum(possibleMinValues)/len(legalActions)


        def maxValue(gameState, depth):
            """
            Function that would get a maximum action value for pacman
            """
            
            pacmanIndex = 0
            legalActions = gameState.getLegalActions(pacmanIndex)

            # If no legal actions are possible, or we have reached the maximum depth possible, return the evaluation for the current gameState
            if depth == self.depth or len(legalActions) == 0:
                return self.evaluationFunction(gameState)


            possibleMaximumValues = []
            
            # Iterate through all possible actions and calculate their values
            for action in legalActions:
                possibleMaximumValues.append(expValue(pacmanIndex + 1, gameState.generateSuccessor(pacmanIndex, action), depth + 1))

            # Return the action which would get the maximum value
            return max(possibleMaximumValues)


        # Get all legal actions for pacman
        actionsPacman = gameState.getLegalActions(0)
        
        allActionsPacman = {}
        
        # Iterate through all possible moves for pacman for the 1st agent and add to the depth 
        for action in actionsPacman:
            allActionsPacman[action] = expValue(1, gameState.generateSuccessor(0, action), 1)
            

        # Return the action that gets the best value
        return max(allActionsPacman, key=allActionsPacman.get)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    
    if currentGameState.isWin():
        return 9999999999999
    
    currPos = currentGameState.getPacmanPosition()
    currGhostStates = currentGameState.getGhostStates()
    currScaredTimes = [ghostState.scaredTimer for ghostState in currGhostStates]
    currFood = currentGameState.getFood()    
    
    currGhostDistances = [(currPos[0]-ghost.configuration.pos[0])**2+(currPos[1]-ghost.configuration.pos[1])**2 for ghost in currGhostStates]
    currSumGhostDistances=  sum([-dist if scared_time>0 else dist for dist,scared_time in zip(currGhostDistances, currScaredTimes)])
    
    try:
        currMinGhostDistances = min([dist for dist,scared_time in zip(currGhostDistances, currScaredTimes) if scared_time==0])
    except ValueError:
        currMinGhostDistances = 0
    
    
    currFoodDistScore = [(currPos[0]-food[0])**2+(currPos[1]-food[1])**2 for food in currFood.asList()]
    currMinFoodDistances = min(currFoodDistScore)

    currScore = 0
    
    if len(currentGameState.getCapsules()) > 0:
        currCapsuleDistScore = [(currPos[0]-capsule[0])**2+(currPos[1]-capsule[1])**2 for capsule in currentGameState.getCapsules()]
        currMinCapsuleFoodDistances = min(currCapsuleDistScore)
    
        currScore -= currMinCapsuleFoodDistances * 0.5
    
    if max(currScaredTimes)>0:
        currScore +=10
    
    for ghostState in currentGameState.getGhostStates():
        if ghostState.scaredTimer > 0 and currPos == ghostState.configuration.pos:
            currScore+=50
        
    currScore += currentGameState.getScore()
    
    for state in currGhostStates:
        if state.getPosition() == currPos and state.scaredTimer == 1:
            return -9999999999
        
    if currentGameState.getCapsules():
        capsuleDistance = [util.manhattanDistance(currPos, capsule) for capsule in currentGameState.getCapsules()]
        nearestCapsule = min(capsuleDistance)
        currScore += float(1 / nearestCapsule)

    currScore += float(1 / currMinFoodDistances) 

    currScore = currScore - currMinGhostDistances * 10 - currSumGhostDistances * 0.1 - currMinFoodDistances
    currScore -= len(currFood.asList()) * 10
    

    return currScore


# Abbreviation
better = betterEvaluationFunction
