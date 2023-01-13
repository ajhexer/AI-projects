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
        score = 0
        food_dist = [util.manhattanDistance(newPos, food) for food in newFood.asList()]
        if len(food_dist) > 0:
            score += 1.0 / min(food_dist)
        else:
            score += 1
        ghost_dist = [util.manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        for dist, time in zip(ghost_dist, newScaredTimes):
            if time > 0:
                score += dist
            else:
                if dist <= 1:
                    score -= 100
                else:
                    score += 1/dist
        score += successorGameState.getScore()
        return score

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
        actions = gameState.getLegalActions(0)
        currentScore = -999
        returnAction = ''
        numberOfGhosts = gameState.getNumAgents() - 1

        def min_f(gameState, depth, agentIndex):
            value = 999

            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            actions = gameState.getLegalActions(agentIndex)

            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)

                if agentIndex == numberOfGhosts:
                    value = min(value, max_f(successor, depth))

                else:
                    value = min(value, min_f(successor, depth, agentIndex + 1))

            return value

        def max_f(gameState, depth):
            value = -999
            currentDepth = depth + 1

            if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
                return self.evaluationFunction(gameState)

            actions = gameState.getLegalActions(0)

            for action in actions:
                successor = gameState.generateSuccessor(0, action)
                value = max(value, min_f(successor, currentDepth, 1))

            return value

        for action in actions:
            nextState = gameState.generateSuccessor(0, action)

            # Next level = min.
            score = min_f(nextState, 0, 1)

            if score > currentScore:
                returnAction = action
                currentScore = score

        return returnAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        alpha = -999
        beta = 999
        actions = gameState.getLegalActions(0)
        currentScore = -999
        returnAction = ''
        numberOfGhosts = gameState.getNumAgents() - 1

        def min_f(gameState, depth, agentIndex, alpha, beta):
            value = 999

            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            actions = gameState.getLegalActions(agentIndex)

            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)

                if agentIndex == numberOfGhosts:
                    value = min(value, max_f(successor, depth, alpha, beta))

                else:
                    value = min(value, min_f(successor, depth, agentIndex + 1, alpha, beta))

                if value < alpha:
                    return value

                beta = min(beta, value)

            return value

        def max_f(gameState, depth, alpha, beta):
            value = -999
            currentDepth = depth + 1

            if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
                return self.evaluationFunction(gameState)

            actions = gameState.getLegalActions(0)

            for action in actions:
                successor = gameState.generateSuccessor(0, action)
                value = max(value, min_f(successor, currentDepth, 1, alpha, beta))

                if value > beta:
                    return value

                alpha = max(alpha, value)

            return value

        for action in actions:
            nextState = gameState.generateSuccessor(0, action)
            score = min_f(nextState, 0, 1, alpha, beta)

            if score > currentScore:
                returnAction = action
                currentScore = score

            if score > beta:
                return returnAction

            alpha = max(alpha, score)

        return returnAction


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
        def max_f(gameState, depth):
            currentDepth = depth + 1

            if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
                return self.evaluationFunction(gameState)

            value = -999
            actions = gameState.getLegalActions(0)

            for action in actions:
                successor= gameState.generateSuccessor(0, action)
                value = max (value, expectedLevel(successor, currentDepth, 1))

            return value


        def expectedLevel(gameState, depth, agent_index):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            actions = gameState.getLegalActions(agent_index)
            totalValue = 0
            actions_num = len(actions)

            for action in actions:
                successor= gameState.generateSuccessor(agent_index, action)

                if agent_index == (gameState.getNumAgents() - 1):
                    expectedvalue = max_f(successor, depth)
                else:
                    expectedvalue = expectedLevel(successor, depth, agent_index + 1)

                totalValue = totalValue + expectedvalue

            if actions_num == 0:
                return  0

            return float(totalValue) / float(actions_num)


        actions = gameState.getLegalActions(0)
        currentScore = -999
        returnAction = ''

        for action in actions:
            nextState = gameState.generateSuccessor(0, action)
            score = expectedLevel(nextState, 0, 1)

            if score > currentScore:
                returnAction = action
                currentScore = score

        return returnAction

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    Don't forget to use pacmanPosition, foods, scaredTimers, ghostPositions!
    DESCRIPTION: <write something here so we know what you did>
    """

    pacmanPosition = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimers = [ghostState.scaredTimer for ghostState in ghostStates]
    ghostPositions = currentGameState.getGhostPositions()

    score = 0
    food_distances = [manhattanDistance(pacmanPosition, food) for food in foods.asList()]
    ghost_distances = [manhattanDistance(pacmanPosition, ghost) for ghost in ghostPositions]

    capsules_num = len(currentGameState.getCapsules())

    eatenFood = len(foods.asList(False))
    total_time_scared = sum(scaredTimers)
    total_ghost_distance = sum(ghost_distances)
    food_distance = 0

    if sum(food_distances) > 0:
        food_distance = 1.0 / sum(food_distances)

    score += currentGameState.getScore() + food_distance + eatenFood

    if total_time_scared > 0:
        score += total_time_scared + (-1 * capsules_num) + (-1 * total_ghost_distance)
    else:
        score += total_ghost_distance + capsules_num

    return score




# Abbreviation
better = betterEvaluationFunction
