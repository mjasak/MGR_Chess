import numpy as np
import config
import logging as lg

class Node:

    def __init__(self, state):
        self.state = state
        self.playerTurn = state.playerTurn
        self.id = state.id
        self.edges = []

    def isLeaf(self):
        if len(self.edges) > 0:
            return False
        else:
            return True


class Edge():

    def __init__(self, inNode, outNode, prior, action):
        self.id = inNode.state.id + '|' + outNode.state.id
        self.inNode = inNode
        self.outNode = outNode
        self.playerTurn = inNode.state.playerTurn
        self.action = action

        self.stats = {
            'N': 0,
            'W': 0,
            'Q': 0,
        }


class MCTS:

    def __init__(self, root, cpuct):
        self.root = root
        self.tree = {}
        self.cpuct = cpuct
        self.addNode(root)

    def __len__(self):
        return len(self.tree)

    def moveToLeaf(self):

        breadcrumbs = []
        currentNode = self.root
        # print(len(currentNode.edges), end=',')

        done = 0
        value = 0
        action_history = []
        while not currentNode.isLeaf():

            maxQU = -99999

            if currentNode == self.root:
                epsilon = config.EPSILON
                nu = np.random.dirichlet([config.ALPHA] * len(currentNode.edges))
            else:
                epsilon = 0
                nu = [0] * len(currentNode.edges)

            Nb = 0
            for action, edge in currentNode.edges:
                Nb = Nb + edge.stats['N']

            for idx, (action, edge) in enumerate(currentNode.edges):
                #
                # U = self.cpuct * \
                #     ((1 - epsilon) * edge.stats['P'] + epsilon * nu[idx]) * \
                #     np.sqrt(Nb) / (1 + edge.stats['N'])
                # if edge.stats['N'] == 0:
                #     nj = 0.0001
                # else:
                #     nj = edge.stats['N']
                # if Nb == 0:
                #     Nb = 0.0001
                nj = edge.stats['N']
                U = self.cpuct * np.sqrt(2*np.log(1+Nb)/(1+nj))
                # print("log:", np.log(1+Nb), "nj: ", nj)
                Q = edge.stats['Q']
                print(U,Q)

                if Q + U > maxQU:
                    maxQU = Q + U
                    simulationAction = action
                    simulationEdge = edge

            action_history.append(simulationAction)
            # lg.logger_mcts.info('action with highest Q + U...%d', simulationAction)

            newState, value, done = currentNode.state.takeAction_s(
                simulationAction)  # the value of the newState from the POV of the new playerTurn
            currentNode = simulationEdge.outNode
            breadcrumbs.append(simulationEdge)
            if action_history.count(simulationAction) > 15:
                # lg.logger_mcts.info('DONE CONDITIONALLY')
                return currentNode, value, done, breadcrumbs
        # lg.logger_mcts.info('DONE...%d', done)

        return currentNode, value, done, breadcrumbs

    def backFill(self, leaf, value, breadcrumbs):

        currentPlayer = leaf.state.playerTurn

        for edge in breadcrumbs:
            playerTurn = edge.playerTurn
            if playerTurn == currentPlayer:
                direction = 1
            else:
                direction = -1

            edge.stats['N'] = edge.stats['N'] + 1
            edge.stats['W'] = edge.stats['W'] + value * direction
            edge.stats['Q'] = edge.stats['W'] / edge.stats['N']

            # lg.logger_mcts.info('updating edge with value %f for player %d... N = %d, W = %f, Q = %f'
            #                     , value * direction
            #                     , playerTurn
            #                     , edge.stats['N']
            #                     , edge.stats['W']
            #                     , edge.stats['Q']
            #                     )
            #
            # edge.outNode.state.render(lg.logger_mcts)

    def addNode(self, node):
        self.tree[node.id] = node

