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
import random
import util

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
    chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    # with food
    foodDistances = []
    foodLeft = 0
    for i in range(newFood.width):
      for j in range(newFood.height):
        if newFood[i][j] == True:
          foodLeft += 1
          d = manhattanDistance(newPos, (i, j))
          foodDistances.append(d)
    # with ghost
    ghostDistances = []
    newGhostPos = [g.getPosition() for g in newGhostStates]
    ghostIsNear = 0
    for g in newGhostPos:
      d = manhattanDistance(newPos, g)
      if d <= 1:
        ghostIsNear += 1
      ghostDistances.append(d)
    closestFood = min(foodDistances) if foodDistances else 0
    closestGhost = min(ghostDistances) if ghostDistances else 0
    foodEval = 1 / closestFood if closestFood != 0 else 0
    ghostEval = 1 / closestGhost if closestGhost != 0 else 0
    isScared = 1 if newScaredTimes[0] != 0 else -1
    # capsules
    capsules = currentGameState.getCapsules()
    capsulesLeft = len(capsules)

    return successorGameState.getScore() + 2 * foodEval + 4 * foodLeft + capsulesLeft + isScared * ghostEval + isScared * ghostIsNear


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

  def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
    self.index = 0  # Pacman is always agent index 0
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

    return self.maxLayer(gameState, 0, 0)[0]

    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

  def miniMax(self, gameState, depth, agentIndex):

    if gameState.isLose() or gameState.isWin() or depth == self.depth * gameState.getNumAgents():
      return self.evaluationFunction(gameState)
    if agentIndex == 0:
      return self.maxLayer(gameState, depth, agentIndex)[1]
    else:
      return self.minLayer(gameState, depth, agentIndex)[1]

  def maxLayer(self, gameState, depth, agentIndex):

    actions = gameState.getLegalActions(agentIndex)
    maxValue = float("-inf")
    currentBestAction = (Directions.STOP, maxValue)
    for action in actions:
      successorGameState = gameState.generateSuccessor(agentIndex, action)
      newDepth = depth + 1
      newAgentIndex = newDepth % gameState.getNumAgents()
      successorAction = (action, self.miniMax(successorGameState, newDepth, newAgentIndex))
      currentBestAction = max(currentBestAction, successorAction, key=lambda x: x[1])
    return currentBestAction

  def minLayer(self, gameState, depth, agentIndex):
    actions = gameState.getLegalActions(agentIndex)
    minValue = float("inf")
    currentBestAction = (Directions.STOP, minValue)
    for action in actions:
      successorGameState = gameState.generateSuccessor(agentIndex, action)
      newDepth = depth + 1
      newAgentIndex = newDepth % gameState.getNumAgents()
      successorAction = (action, self.miniMax(successorGameState, newDepth, newAgentIndex))
      currentBestAction = min(currentBestAction, successorAction, key=lambda x: x[1])
    return currentBestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"

    return self.AlphaBetaMaxLayer(gameState, 0, 0, float("-inf"), float("inf"))[0]
    util.raiseNotDefined()

  def AlphaBeta(self, gameState, depth, agentIndex, alpha, beta):

    if gameState.isLose() or gameState.isWin() or depth == self.depth * gameState.getNumAgents():
      return self.evaluationFunction(gameState)
    if agentIndex == 0:
      return self.AlphaBetaMaxLayer(gameState, depth, agentIndex, alpha, beta)[1]
    else:
      return self.AlphaBetaMinLayer(gameState, depth, agentIndex, alpha, beta)[1]

  def AlphaBetaMaxLayer(self, gameState, depth, agentIndex, alpha, beta):

    actions = gameState.getLegalActions(agentIndex)
    maxValue = float("-inf")
    currentBestAction = (Directions.STOP, maxValue)
    for action in actions:
      successorGameState = gameState.generateSuccessor(agentIndex, action)
      newDepth = depth + 1
      newAgentIndex = newDepth % gameState.getNumAgents()
      successorAction = (action, self.AlphaBeta(successorGameState, newDepth, newAgentIndex, alpha, beta))
      currentBestAction = max(currentBestAction, successorAction, key=lambda x: x[1])

      if currentBestAction[1] > beta:
        return currentBestAction
      else:
        alpha = max(alpha, currentBestAction[1])
    return currentBestAction

  def AlphaBetaMinLayer(self, gameState, depth, agentIndex, alpha, beta):
    actions = gameState.getLegalActions(agentIndex)
    minValue = float("inf")
    currentBestAction = (Directions.STOP, minValue)
    for action in actions:
      successorGameState = gameState.generateSuccessor(agentIndex, action)
      newDepth = depth + 1
      newAgentIndex = newDepth % gameState.getNumAgents()
      successorAction = (action, self.AlphaBeta(successorGameState, newDepth, newAgentIndex, alpha, beta))
      currentBestAction = min(currentBestAction, successorAction, key=lambda x: x[1])

      if currentBestAction[1] < alpha:
        return currentBestAction
      else:
        beta = min(beta, currentBestAction[1])
    return currentBestAction


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
    return self.maxLayer(gameState, 0, 0)[0]

    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

  def expectiMax(self, gameState, depth, agentIndex):

    if gameState.isLose() or gameState.isWin() or depth == self.depth * gameState.getNumAgents():
      return self.evaluationFunction(gameState)
    if agentIndex == 0:
      return self.maxLayer(gameState, depth, agentIndex)[1]
    else:
      return self.chanceLayer(gameState, depth, agentIndex)[1]

  def maxLayer(self, gameState, depth, agentIndex):

    actions = gameState.getLegalActions(agentIndex)
    maxValue = float("-inf")
    currentBestAction = (Directions.STOP, maxValue)
    for action in actions:
      successorGameState = gameState.generateSuccessor(agentIndex, action)
      newDepth = depth + 1
      newAgentIndex = newDepth % gameState.getNumAgents()
      successorAction = (action, self.expectiMax(successorGameState, newDepth, newAgentIndex))
      currentBestAction = max(currentBestAction, successorAction, key=lambda x: x[1])
    return currentBestAction

  def chanceLayer(self, gameState, depth, agentIndex):
    actions = gameState.getLegalActions(agentIndex)
    totalScore = 0.0
    action = 'temp'

    for action in actions:
      successorGameState = gameState.generateSuccessor(agentIndex, action)
      newDepth = depth + 1
      newAgentIndex = newDepth % gameState.getNumAgents()
      successorAction = (action, self.expectiMax(successorGameState, newDepth, newAgentIndex))
      totalScore += successorAction[1]
      action = action
    average = totalScore / float(len(actions))
    return (action, average)


def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  currentScore = scoreEvaluationFunction(currentGameState)
  if currentGameState.isWin():
    return float("inf")
  if currentGameState.isLose():
    return -float("inf")

  pacmanPos = currentGameState.getPacmanPosition()

  # food
  food = currentGameState.getFood()
  foodList = currentGameState.getFood().asList()
  foodLeft = currentGameState.getNumFood()
  foodDistances = []
  for food in foodList:
    d = manhattanDistance(pacmanPos, food)
    foodDistances.append(d)
  closestFood = min(foodDistances) if foodDistances else 0

  # capsules
  capsuleDistance = []
  capsules = currentGameState.getCapsules()
  for capsule in capsules:
    d = manhattanDistance(pacmanPos, capsule)
    capsuleDistance.append(d)
  closestCapsule = min(capsuleDistance) if capsuleDistance else 0
  capsuleleft = len(capsules)

  # ghost
  ghostStates = currentGameState.getGhostStates()
  ghostPos = [g.getPosition() for g in ghostStates]
  ghost1, ghost2 = ghostPos[0], ghostPos[1]
  dToG1, dToG2 = manhattanDistance(ghost1, pacmanPos), manhattanDistance(ghost2, pacmanPos)
  closestGhost = min(dToG1, dToG2)
  scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
  ghost1isScared = -1 if scaredTimes[0] != 0 else 0
  ghost2isScared = -1 if scaredTimes[1] != 0 else 0

  return currentScore + \
      -1 * closestFood + \
      -1 * foodLeft + \
      ghost1isScared * dToG1 + \
      ghost2isScared * dToG2 + \
      -35 * capsuleleft


# Abbreviation
better = betterEvaluationFunction
