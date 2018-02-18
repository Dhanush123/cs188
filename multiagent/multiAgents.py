# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
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

        "Add more of your code here if you want to"
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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        scores = successorGameState.getScore()
        ghostPos = newGhostStates[0].getPosition()

        if any(newScaredTimes):
            return float('inf')

        foodsWeight = 5
        ghostsWeight = -20
        food = min([manhattanDistance(newPos,foodPos) for foodPos in newFood]) if len(newFood) else 0
        foods = foodsWeight/food if food > 0 else 0
        ghosts = ghostsWeight/manhattanDistance(newPos, ghostPos) if manhattanDistance(newPos, ghostPos) > 0 else 0
        retval = foods+ghosts+scores
        return retval

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

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """

        global bestAction
        bestAction = -float("inf")
        def minimax(player, ply, isPac, state):
          global bestAction
          if ply == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
          if isPac:
            bestV = -float("inf")
            actions = state.getLegalActions(player)
            for action in actions:
              newState = state.generateSuccessor(player, action)
              nextPlayer = (player+1)%state.getNumAgents()
              v = minimax(nextPlayer,ply,False,newState)
              if v > bestV and ply == self.depth:
                bestAction = action
              bestV = max(bestV, v)
            return bestV
          else:
            bestV = float("inf")
            actions = state.getLegalActions(player)
            for action in actions:
              newState = state.generateSuccessor(player, action)
              nextPlayer = (player+1)%state.getNumAgents()
              isPac = True if nextPlayer == 0 else False
              tempPly = ply-1 if isPac else ply
              v = minimax(nextPlayer,tempPly,isPac,newState)
              bestV = min(bestV, v)
            return bestV 
        finalAction = minimax(0,self.depth,True,gameState)
        # pacTopActions = gameState.getLegalActions(0)
        # pacTopOptions = [minimax(0, self.depth, True, gameState) for action in pacTopActions]
        # finalVal = max(pacTopOptions)
        # topIndex = pacTopActions
        # print "pacTopActions",pacTopActions
        # print "pacTopOptions",pacTopOptions
        # print "-----> finalVal:",finalVal
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        global bestAction
        bestAction = -float("inf")
        def minimax(player, ply, isPac, state, alpha, beta):
          global bestAction
          actions = state.getLegalActions(player)
          if ply == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state) 
          if isPac:
            bestV = -float("inf")
            for action in actions:
              newState = state.generateSuccessor(player, action)
              nextPlayer = (player+1)%state.getNumAgents()
              v = minimax(nextPlayer, ply, False, newState, alpha, beta)
              if v > bestV and ply == self.depth:
                bestAction = action
              bestV = max(bestV, v)
              if bestV > beta:
                return bestV
              alpha = max(alpha, bestV)
            return bestV
          else:
            bestV = float("inf")
            for action in actions:
              newState = state.generateSuccessor(player, action)
              nextPlayer = (player+1)%state.getNumAgents()
              isPac = True if nextPlayer == 0 else False
              tempPly = ply-1 if isPac else ply
              v = minimax(nextPlayer,tempPly,isPac,newState, alpha, beta)
              bestV = min(bestV, v)
              if bestV < alpha:
                return bestV
              beta = min(beta, bestV)
            return bestV 
        finalAction = minimax(0,self.depth,True,gameState, -float("inf"), float("inf"))
        return bestAction

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
        global bestAction
        bestAction = -float("inf")
        def minimax(player, ply, isPac, state):
          global bestAction
          if ply == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
          if isPac:
            bestV = -float("inf")
            actions = state.getLegalActions(player)
            for action in actions:
              newState = state.generateSuccessor(player, action)
              nextPlayer = (player+1)%state.getNumAgents()
              v = minimax(nextPlayer,ply,False,newState)
              if v > bestV and ply == self.depth:
                bestAction = action
              bestV = max(bestV, v)
            return bestV
          else:
            bestV = 0
            actions = state.getLegalActions(player)
            for action in actions:
              newState = state.generateSuccessor(player, action)
              nextPlayer = (player+1)%state.getNumAgents()
              isPac = True if nextPlayer == 0 else False
              tempPly = ply-1 if isPac else ply
              bestV += minimax(nextPlayer,tempPly,isPac,newState)
            return bestV/float(len(actions))
        finalAction = minimax(0,self.depth,True,gameState)
        return bestAction       

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """

    pacPos = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    ghostPos = []
    for g in ghostStates:
      ghostPos.append(g.getPosition())
    foods = currentGameState.getFood().asList()
    scores = currentGameState.getScore()
    
    scaredTime = [ghostState.scaredTimer for ghostState in ghostStates]

    if any(scaredTime):
      return float('inf')

    foodsWeight = 5
    ghostsWeight = -20
    foodScore = foodsWeight/min([manhattanDistance(pacPos,foodPos) for foodPos in foods]) if len(foods) else 0
    ghostScore = 0
    for g in ghostPos:
      ghostScore += ghostsWeight/manhattanDistance(pacPos, g) if manhattanDistance(pacPos, g) > 0 else 0

    return scores + foodScore + ghostScore 

# Abbreviation
better = betterEvaluationFunction
