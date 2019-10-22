# -*- coding: UTF-8 -*-
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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if giàng you want to"

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
        # vị trí người chơi
        newPos = successorGameState.getPacmanPosition()
        # ví trí đò ăn
        newFood = successorGameState.getFood()

        # vị trí ma
        newGhostStates = successorGameState.getGhostStates()
        # thời gian con ma sợ
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        newCapsule = successorGameState.getCapsules()
        # so big dot
        numberOfCapsules = len(newCapsule)

        "*** YOUR CODE HERE ***"
        actions = {'West':0.01,'South':0.02,'East':0.03,'North':0.04,'Stop':-0.01}
        post_to_capsule=1
        if numberOfCapsules>0:
          post_to_capsule = min([util.manhattanDistance(newPos, x) for x in newCapsule])

        newFoodList = newFood.asList()

        max_food_to_food = -1
        min_pos_to_food = -1
        time_scare = 0
        if len(newFoodList) > 0:
            max_food_to_food = max(
                [[[food, food2], util.manhattanDistance(food, food2)] for food in newFoodList for food2 in newFoodList],
                key=lambda t: t[1])
            min_pos_to_food = min([util.manhattanDistance(newPos, max_food_to_food[0][0]),
                                   util.manhattanDistance(newPos, max_food_to_food[0][1])])
        distances_to_ghosts = min(
            [util.manhattanDistance(newPos, ghost_state) for ghost_state in successorGameState.getGhostPositions()])

        if distances_to_ghosts <= 1:
            return -1000;

        time_scare= min(newScaredTimes)

        if time_scare!= 0:
            return 100 / (min_pos_to_food+max_food_to_food[1]) + successorGameState.getScore() + 3 / (numberOfCapsules + 1) - 1 / (
            distances_to_ghosts+1) + 9000 / (len(newFoodList) + 1)+1000 + actions[action]+1/max_food_to_food[1]
        if max_food_to_food != -1:
            return 100 / (min_pos_to_food + max_food_to_food[1]) + successorGameState.getScore() + 300 / (
                    numberOfCapsules + 1) + 9000 / (len(newFoodList) + 1+numberOfCapsules)  + 10/ post_to_capsule+time_scare
        return 100 / (min_pos_to_food) + successorGameState.getScore() + 3 / (numberOfCapsules + 1)  + 9000 / (len(newFoodList) + 1+numberOfCapsules) + 10/ post_to_capsule+time_scare


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
        "*** YOUR CODE HERE ***"
        root_value = self.value(gameState, 0, self.index)

        action = root_value[1]

        return action

    def value(self, gameState, current_depth, agent_index):

        if agent_index == gameState.getNumAgents():
            current_depth = current_depth + 1
            agent_index = 0

        legal_action = gameState.getLegalActions(agent_index)
        if len(legal_action) == 0:
            eval_value = self.evaluationFunction(gameState)
            return [eval_value]

        if current_depth == self.depth:
            eval_value = self.evaluationFunction(gameState)
            return [eval_value]

        if agent_index == 0:
            return self.max_value(gameState, current_depth, agent_index)
        else:
            return self.min_value(gameState, current_depth, agent_index)

    def max_value(self, gameState, current_depth, agent_index):
        node_value = [-float("inf")]
        action_possible = gameState.getLegalActions(agent_index)

        for action in action_possible:
            successor_state = gameState.generateSuccessor(agent_index, action)

            successor_evalvalue = self.value(successor_state, current_depth, agent_index + 1)

            successor_evalvalue = successor_evalvalue[0]

            if (successor_evalvalue >= node_value[0]):
                node_value = [successor_evalvalue, action]

        return node_value

    def min_value(self, gameState, current_depth, agent_index):

        node_value = [float("inf")]

        action_list = gameState.getLegalActions(agent_index)
        print (action_list)

        for action in action_list:
            successor_state = gameState.generateSuccessor(agent_index, action)
            successor_evalvalue = self.value(successor_state, current_depth, agent_index + 1)

            successor_evalvalue = successor_evalvalue[0]

            if (successor_evalvalue <= node_value[0]):
                node_value = [successor_evalvalue, action]

        return node_value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        root_value = self.value(gameState, 0, self.index, -float("inf"), float("inf"))

        action = root_value[1]

        return action

    def value(self, gameState, current_depth, agent_index, alpha, beta):

        if agent_index == gameState.getNumAgents():
            current_depth = current_depth + 1
            agent_index = 0

        if current_depth == self.depth:
            eval_value = self.evaluationFunction(gameState)
            return [eval_value]

        legal_action = []
        legal_action = gameState.getLegalActions(agent_index)

        if len(legal_action) == 0:
            eval_value = self.evaluationFunction(gameState)
            return [eval_value]

        if agent_index == 0:
            return self.max_value(gameState, current_depth, agent_index, alpha, beta)
        else:
            return self.min_value(gameState, current_depth, agent_index, alpha, beta)

    def max_value(self, gameState, current_depth, agent_index, alpha, beta):

        node_value = [-float("inf")]
        action_list = gameState.getLegalActions(agent_index)

        for action in action_list:
            successor_state = gameState.generateSuccessor(agent_index, action)
            successor_evalvalue = self.value(successor_state, current_depth, agent_index + 1, alpha, beta)

            successor_evalvalue = successor_evalvalue[0]

            if (successor_evalvalue >= node_value[0]):
                node_value = [successor_evalvalue, action]

            max_value = node_value[0]

            if max_value > beta:
                return node_value

            alpha = max(max_value, alpha)

        return node_value

    def min_value(self, gameState, current_depth, agent_index, alpha, beta):

        node_value = [float("inf")]
        action_list = gameState.getLegalActions(agent_index)

        for action in action_list:
            successor_state = gameState.generateSuccessor(agent_index, action)
            successor_evalvalue = self.value(successor_state, current_depth, agent_index + 1, alpha, beta)

            successor_evalvalue = successor_evalvalue[0]

            if (successor_evalvalue <= node_value[0]):
                node_value = [successor_evalvalue, action]

            min_value = node_value[0]

            if min_value < alpha:
                return node_value

            beta = min(min_value, beta)

        return node_value


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

        def expectimax(agent, depth, gameState):
            if gameState.isLose() or gameState.isWin() or depth == self.depth:
                return self.evaluationFunction(gameState)
            if agent == 0:
                return max(expectimax(1, depth, gameState.generateSuccessor(agent, newState)) for newState in
                           gameState.getLegalActions(agent))
            else:  # performing expectimax action for ghosts/chance nodes.
                nextAgent = agent + 1  # calculate the next agent and increase depth accordingly.
                if gameState.getNumAgents() == nextAgent:
                    nextAgent = 0
                if nextAgent == 0:
                    depth += 1
                return sum(expectimax(nextAgent, depth, gameState.generateSuccessor(agent, newState)) for newState in
                           gameState.getLegalActions(agent)) / float(len(gameState.getLegalActions(agent)))

        """Performing maximizing task for the root node i.e. pacman"""
        maximum = float("-inf")
        action = Directions.WEST

        for agentState in gameState.getLegalActions(0):
            utility = expectimax(1, 0, gameState.generateSuccessor(0, agentState))
            if utility > maximum or maximum == float("-inf"):
                maximum = utility
                action = agentState

        return action


def betterEvaluationFunction(currentGameState):
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()

    newFoodList = newFood.asList()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    newCapsule = currentGameState.getCapsules()

    # so big dot
    numberOfCapsules = len(newCapsule)
    max_food_to_food = -1
    min_pos_to_food = -1
    time_scare = 0
    actions = {'West': 0.01, 'South': 0.02, 'East': 0.03, 'North': 0.04, 'Stop': -0.01}
    post_to_capsule = 1
    if numberOfCapsules > 0:
        post_to_capsule = min([util.manhattanDistance(newPos, x) for x in newCapsule])

    newFoodList = newFood.asList()

    max_food_to_food = -1
    min_pos_to_food = -1
    time_scare = 0
    if len(newFoodList) > 0:
        max_food_to_food = max(
            [[[food, food2], util.manhattanDistance(food, food2)] for food in newFoodList for food2 in newFoodList],
            key=lambda t: t[1])
        min_pos_to_food = min([util.manhattanDistance(newPos, max_food_to_food[0][0]),
                               util.manhattanDistance(newPos, max_food_to_food[0][1])])
    distances_to_ghosts = min(
        [util.manhattanDistance(newPos, ghost_state) for ghost_state in currentGameState.getGhostPositions()])

    if distances_to_ghosts <= 1:
        return -1000;

    time_scare = min(newScaredTimes)

    if time_scare != 0:
        if max_food_to_food != -1:
         return 100 / (min_pos_to_food + max_food_to_food[1]) + currentGameState.getScore()  - 1 / (
                       distances_to_ghosts + 1) + 9000 / (len(newFoodList) + 1) + 1000 +   1 / \
                (max_food_to_food[1]+1)+random.random()
         return 100 / (min_pos_to_food + max_food_to_food[1]) + currentGameState.getScore() + - 1 / (
                       distances_to_ghosts + 1) + 9000 / (len(newFoodList) + 1) + 1000 +  +random.random()
    if max_food_to_food != -1:
        return 100 / (min_pos_to_food + max_food_to_food[1]) + currentGameState.getScore() + 300 / (
                3*numberOfCapsules + 1) + 9000 / (
                           len(newFoodList) + 1 + numberOfCapsules) + 10 / post_to_capsule + time_scare-1/distances_to_ghosts
    return 100 / (min_pos_to_food) + currentGameState.getScore()  + 9000 / (
                len(newFoodList) + 1 + 3*numberOfCapsules) + 10 / post_to_capsule + time_scare -1/distances_to_ghosts


# Abbreviation
better = betterEvaluationFunction
