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
        '''
#         "Add more of your code here if you want to"
#         print " Current position: ", gameState.getPacmanPosition()
#         print " Available: ", [( index,legalMoves[index], scores[index]) for index in range(len(scores))]
#         print " ChosenIndex", chosenIndex
#         print " Action ", legalMoves[chosenIndex]
        '''
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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
        import math
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        
#         float("inf") serves here as a temporary MAX number , same goes for -float("inf")        
        minimumFood = float("inf")
        tempFood = successorGameState.getFood().asList()
        
#         gets the other food as a list and calculates the minimum manhattan distance between the position and the food 
        for food in tempFood:
            minimumFood = min(minimumFood, manhattanDistance(newPos, food))

#          if theres a ghost nearby set return a negative number because they will eat us           
        for ghost in successorGameState.getGhostPositions():
            if (manhattanDistance(newPos, ghost) < 2):
                return -float('inf')

        return successorGameState.getScore() + 1.0/minimumFood


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

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        
        
        
#         We are gonna use the "IsWin" and "IsLose" functions from pacman.py to identify if a current state is a win or a loss.

#         starting position 
#         v - maxvalue (state, 
        v = (float('-inf'), "stop")
        agentIndex = 1
        for action in gameState.getLegalActions(0):
            v = max(v, (self.minimax(gameState.generateSuccessor(0, action), agentIndex, self.depth), action))
        return v[1]

    def minimax(self, currentGameState, agentIndex, depth):
        if agentIndex == currentGameState.getNumAgents():
            agentIndex = 0
            depth -= 1
        if depth == 0 or currentGameState.isWin() or currentGameState.isLose():
            return self.evaluationFunction(currentGameState)        
        if(agentIndex == 0):
            v = float('-inf')
            for action in currentGameState.getLegalActions(0):
                v = max(v, self.minimax(currentGameState.generateSuccessor(0, action), agentIndex + 1, depth))
            return v
        else:
            v = float('inf')
            for action in currentGameState.getLegalActions(agentIndex):
                v = min(v, self.minimax(currentGameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth))
            return v

    """return self.maxval(gameState, 0, 0)[0]

    def minimax(self, gameState, agentIndex, depth):
        if depth is self.depth * gameState.getNumAgents() \
                or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        if agentIndex is 0:
            return self.maxval(gameState, agentIndex, depth)[1]
        else:
            return self.minval(gameState, agentIndex, depth)[1]

    def maxval(self, gameState, agentIndex, depth):
        bestAction = ("max",-float("inf"))
        for action in gameState.getLegalActions(agentIndex):
            succAction = (action,self.minimax(gameState.generateSuccessor(agentIndex,action),
                                      (depth + 1)%gameState.getNumAgents(),depth+1))
            bestAction = max(bestAction,succAction,key=lambda x:x[1])
        return bestAction

    def minval(self, gameState, agentIndex, depth):
        bestAction = ("min",float("inf"))
        for action in gameState.getLegalActions(agentIndex):
            succAction = (action,self.minimax(gameState.generateSuccessor(agentIndex,action),
                                      (depth + 1)%gameState.getNumAgents(),depth+1))
            bestAction = min(bestAction,succAction,key=lambda x:x[1])
        return bestAction
    """
        
        
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"        
        v = (float('-inf'), "stop")
        a, b = float('-inf'), float('inf')
        agent = 1
        for action in gameState.getLegalActions(0):
            newv= self.alphaBeta(gameState.generateSuccessor(0, action), agent, self.depth, a, b)
            v = max(v, (newv, action))
            if v[0] > b: return v[1]
            a = max(a, v[0])
        return v[1]


    def alphaBeta(self, currentGameState, agent, depth, a, b):
        if agent == currentGameState.getNumAgents():
            agent = 0
            depth -= 1
        if depth == 0 or currentGameState.isWin() or currentGameState.isLose():
            return self.evaluationFunction(currentGameState)     
        if(agent == 0):
            v = float('-inf')
            for action in currentGameState.getLegalActions(0):
                newv = self.alphaBeta(currentGameState.generateSuccessor(0, action), agent + 1, depth, a, b)
                v = max(v, newv)
                if v > b: 
                    return v
                a = max(a, v)
            return v
        else:
            v = float('inf')
            for action in currentGameState.getLegalActions(agent):
                newv = self.alphaBeta(currentGameState.generateSuccessor(agent, action), agent + 1, depth, a, b)
                v = min(v, newv)
                if v < a: 
                    return v
                b = min(b, v)
            return v

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
        v = (float('-inf'), "stop")
        agent = 1
        for action in gameState.getLegalActions(0):
            v = max(v, (self.expectimax(gameState.generateSuccessor(0, action), agent, self.depth), action))
        return v[1]  


    def expectimax(self, currentGameState, agent, depth):
        if agent == currentGameState.getNumAgents():
            agent = 0
            depth -= 1
        if depth == 0 or currentGameState.isWin() or currentGameState.isLose():
            #print "terminal ", self.evaluationFunction(currentGameState)
            return self.evaluationFunction(currentGameState)        
        if(agent == 0):
            v = float('-inf')
            for action in currentGameState.getLegalActions(0):
                v = max(v, self.expectimax(currentGameState.generateSuccessor(0, action), agent + 1, depth))
            #print currentGameState.state , "returns", v
            return v
        else:
            #v = float('inf')
            v = 0
            p = 1.0/len(currentGameState.getLegalActions(agent))
            for action in currentGameState.getLegalActions(agent):
                v = v + p * self.expectimax(currentGameState.generateSuccessor(agent, action), agent + 1, depth)  
            return v


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION:
    safety: the bigger the value, the safer pacman is
            values: [-110,max_scareTimer]
    target: the bigger the value, the closer pacman is to a food
            values: (0,100]
    mobility: number of available actions
            values: [1,4]
    hunt: distance from closest ghost
            values: 0 if ghosts are not scared
                    (~1,20)
    """
    "*** YOUR CODE HERE ***"
    currentGameState
    Pos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    hunt = 0
    
    if Food.count() == 0 or Pos in Food:
        ClosestFood = 0.1
    else:
        ClosestFood = min([(util.manhattanDistance(Pos, x)) for x in Food.asList()])
    ClosestDistToG, timer = min([(util.manhattanDistance(Pos, x.getPosition()), x.scaredTimer) for x in GhostStates])
    
    # weights for 
    #safety, target, foodcount, mobility, hunt
    w1, w2, w3, w4, w5 = 1.1, 1, 0.6, 0.7, 1.2

    if timer != 0:
        safety = timer
        if timer > ClosestDistToG:
            hunt = 20.0/ClosestDistToG
            w2 = 0.3
        elif ClosestDistToG == 0:
            hunt = 100
    elif (not ClosestDistToG): #really close to ghost
        safety = -110
    else:
        safety = -10.0/ClosestDistToG
    target = 10.0/ClosestFood

    mobility = len(currentGameState.getLegalActions())
    
    return w1*safety + w2*target + w3*Food.count() + w4*mobility + w5*hunt + currentGameState.getScore()

# Abbreviation
better = betterEvaluationFunction

