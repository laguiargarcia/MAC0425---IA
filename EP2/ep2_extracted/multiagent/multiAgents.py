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

# Lucas Aguiar Garcia - NUSP 13672770


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

    def __init__(self, evalFn = 'scoreEvaluationFunction'):
        self.evaluationFunction = util.lookup(evalFn, globals())

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
        successors = [(gameState.generatePacmanSuccessor(action)) for action in legalMoves]
        scores = [self.evaluationFunction(succesor) for successors in successors]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return legalMoves[chosenIndex]

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
    Your minimax agent (question 1)
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
        numAgents = gameState.getNumAgents()

        def value(state, agentIndex, depth):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            if agentIndex == 0 and depth == self.depth:
                return self.evaluationFunction(state)
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth
            scores = [value(state.generateSuccessor(agentIndex, action), nextAgent, nextDepth)
                      for action in state.getLegalActions(agentIndex)]
            return max(scores) if agentIndex == 0 else min(scores)

        nextAgent = (1) % numAgents
        nextDepth = 1 if nextAgent == 0 else 0
        bestScore = float('-inf')
        bestAction = None
        for action in gameState.getLegalActions(0):
            score = value(gameState.generateSuccessor(0, action), nextAgent, nextDepth)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        numAgents = gameState.getNumAgents()

        def value(state, agentIndex, depth, alpha, beta):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            if agentIndex == 0 and depth == self.depth:
                return self.evaluationFunction(state)
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth
            if agentIndex == 0:
                best = float('-inf')
                for action in state.getLegalActions(agentIndex):
                    best = max(best, value(state.generateSuccessor(agentIndex, action),
                                           nextAgent, nextDepth, alpha, beta))
                    if best > beta:
                        return best
                    alpha = max(alpha, best)
                return best
            best = float('inf')
            for action in state.getLegalActions(agentIndex):
                best = min(best, value(state.generateSuccessor(agentIndex, action),
                                       nextAgent, nextDepth, alpha, beta))
                if best < alpha:
                    return best
                beta = min(beta, best)
            return best

        alpha = float('-inf')
        beta = float('inf')
        nextAgent = (1) % numAgents
        nextDepth = 1 if nextAgent == 0 else 0
        bestScore = float('-inf')
        bestAction = None
        for action in gameState.getLegalActions(0):
            score = value(gameState.generateSuccessor(0, action), nextAgent, nextDepth, alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action
            alpha = max(alpha, bestScore)
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        numAgents = gameState.getNumAgents()

        def value(state, agentIndex, depth):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            if agentIndex == 0 and depth == self.depth:
                return self.evaluationFunction(state)
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth
            scores = [value(state.generateSuccessor(agentIndex, action), nextAgent, nextDepth)
                      for action in state.getLegalActions(agentIndex)]
            if agentIndex == 0:
                return max(scores)
            return sum(scores) / len(scores)

        nextAgent = (1) % numAgents
        nextDepth = 1 if nextAgent == 0 else 0
        bestScore = float('-inf')
        bestAction = None
        for action in gameState.getLegalActions(0):
            score = value(gameState.generateSuccessor(0, action), nextAgent, nextDepth)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction

def betterEvaluationFunction(currentGameState: GameState):
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()
    score = currentGameState.getScore()
    # Incentive to get food
    if food:
        # Minimize distanace to closest food
        minFoodDist = min(manhattanDistance(pos, f) for f in food)
        score += 10.0 / (minFoodDist + 1)
        # Minimize remaining food
        score -= 4 * len(food)
    # Avoid ghosts
    for ghost in ghosts:
        ghostPos = ghost.getPosition()
        dist = manhattanDistance(pos, ghostPos)
        # Get closer to scared ghosts
        if ghost.scaredTimer > 0:
            score += 20.0 / (dist + 1)
        else:
            if dist <= 1:
                # Avoid immediate death
                score -= 500
            else:
                # Get away from ghosts
                score -= 2.0 / dist
    return score

# Abbreviation
better = betterEvaluationFunction
