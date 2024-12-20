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
        "Add more of your code here if you want to"
        print " Current position: ", gameState.getPacmanPosition()
        print " Available: ", [( index,legalMoves[index], scores[index]) for index in range(len(scores))]
        print " ChosenIndex", chosenIndex
        print " Action ", legalMoves[chosenIndex]
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
        # if newPos in currentGameState.getFood().asList():
        #   ClosestFood = 0.1
        # else:
        #   ClosestFood = min([(util.manhattanDistance(newPos,x)) for x in newFood.asList()])
        # ClosestDistToG = min([(util.manhattanDistance(newPos,x.getPosition())) for x in newGhostStates])

        # if(not ClosestDistToG): #really close to ghost
        #   safety = -110
        # else:
        #   safety = -10.0/ClosestDistToG
        # target = 10.0/ClosestFood

        # mobility = len(successorGameState.getLegalActions())

        # w1, w2, w3, w4 = 1, 1, 0, 0.7
        # return w1*safety + w2*target + w3*newFood.count() + w4*mobility + successorGameState.getScore()
        minFood = float("inf")
        newFood = successorGameState.getFood().asList()

        for food in newFood:
            minFood = min(minFood, manhattanDistance(newPos, food))

        for ghost in successorGameState.getGhostPositions():
            if (manhattanDistance(newPos, ghost) < 2):
                return -float('inf')

        return successorGameState.getScore() + 1.0/minFood
        #return int(round(ClosestDistToG/ClosestFood))


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
        # We are gonna use the "IsWin" and "IsLose" functions from pacman.py to identify if a current state is a win or a loss.
        # starting position 
        # v - maxvalue (state, 
        v = (float('-inf'), "stop")
        agentIndex = 1
        # get all actions for start state and find the max 
        for action in gameState.getLegalActions(0):
            v = max(v, (self.minimax(gameState.generateSuccessor(0, action), agentIndex, self.depth), action))
        return v[1]

    # each time we go to the next level of the game's search tree we decrease the depth
    # depth = 0 -> end of tree so we call the evaluation function for the state
    # reminder = float("inf") and -float("inf") are max min numbers for comparison
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



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"    

        """
        ab algorithm (diafaneies)
        function absearch(state)  returns an action 
        v<- maxvalue(state, -inf, +inf)
        return the action in actions(state) with value v

        function maxvalue(state,a,b) returns a utility value
            if state.is win or state.is lose return utility(State) or depth =0
            v<- -inf
            for each a in currentGameState.getLegalActions(0) do 
                v <- max(v,minvalue(result(state,a) , a, b))
                if(v>= b ) return v
                a <- max (a,v)
            return v

        function minvalue(state,a,b) returns a utility value
            if state.is win or state.is lose return utility(State) or depth =0
            v<- +inf
            for each a in currentGameState.getLegalActions(0) do 
                v <- max(v,maxvalue(result(state,a) , a, b))
                if(v<= b ) return v
                a <- max (b,v)
            return
        """
        # a = best value for max
        # b = best value for min 
        v = (float('-inf'), "stop")
        a, b = float('-inf'), float('inf')
        agent = 1
        for action in gameState.getLegalActions(0):
            tempV= self.abpruning(gameState.generateSuccessor(0, action), agent, self.depth, a, b)
            v = max(v, (tempV, action))
            if v[0] > b: return v[1]
            a = max(a, v[0])
        return v[1]


    def abpruning(self, currentGameState, agent, depth, a, b):
        if agent == currentGameState.getNumAgents():
            agent = 0
            depth -= 1

        if depth == 0 or currentGameState.isWin() or currentGameState.isLose():
            return self.evaluationFunction(currentGameState)     

        if(agent == 0):
            v = float('-inf')
            for action in currentGameState.getLegalActions(0):
                tempV = self.abpruning(currentGameState.generateSuccessor(0, action), agent + 1, depth, a, b)
                v = max(v, tempV)
                if v > b: 
                    return v
                a = max(a, v)
            return v
        else:
            v = float('inf')
            for action in currentGameState.getLegalActions(agent):
                tempV = self.abpruning(currentGameState.generateSuccessor(agent, action), agent + 1, depth, a, b)
                v = min(v, tempV)
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


        ''' expectiminimax algorith (diafaneies)
            if player is max :  
                maxexpectiminimax(result(s,a)) 
            else if player is min :  
                minexpectiminimax(result(s,a)) 
            else player is chance :  
                sum(p(r) * expectiminimax(result(s,a)) )
            https://en.wikipedia.org/wiki/Expectiminimax

        '''


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
            return self.evaluationFunction(currentGameState)   

        if(agent == 0):
            v = float('-inf')
            for action in currentGameState.getLegalActions(0):
                v = max(v, self.expectimax(currentGameState.generateSuccessor(0, action), agent + 1, depth))
            return v
        else:
            v = 0
            # p = probability of r 
            # r = result of throwing dice or just a random event 
            p = 1.0/len(currentGameState.getLegalActions(agent))
            for action in currentGameState.getLegalActions(agent):
                v = v + p * self.expectimax(currentGameState.generateSuccessor(agent, action), agent + 1, depth)  
            return v

"""

As for your reflex agent evaluation function,
you may want to use the reciprocal of important values (such as distance to food) 
rather than the values themselves.

Important values can be : 
    safety of pacman at a certain position
    food available in the maze
    distance from the closest ghost to our position
    how many actions are available to do 
    what is the target , food , ghost or dot

One way you might want to write your evaluation function is to use a linear combination of features. 
That is, compute values for features about the state that you think are important,
and then combine those features by multiplying them by different values and adding the results together.
You might decide what to multiply each feature by based on how important you think it is.
"""

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

