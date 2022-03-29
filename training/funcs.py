import numpy as np
import random
# import pandas as pd
import loggers as lg

from game import Game, GameState
from model import Residual_CNN

from agent import Agent, User
from datetime import datetime
import config

# path_to_movelist = r"C:\Users\jasak\PycharmProjects\pythonProject\movelist.txt"
import moves

# def movelist() -> np.ndarray:
#     return np.squeeze(pd.read_csv(path_to_movelist, delimiter='\n', header=None).values)
#


def progress(percent=0, width=40):
    # The number of hashes to show is based on the percent passed in. The
    # number of blanks is whatever space is left after.
    hashes = width * percent // 100
    blanks = width - hashes
    print('\r[', hashes*'#', blanks*' ', ']', f' {percent:.0f}%', sep='',
        end='\t', flush=True)


MAX_TURNS = config.MAX_TURNS


def playMatchesBetweenVersions(env, run_version, player1version, player2version, EPISODES, logger, turns_until_tau0,
                               goes_first=0):
    if player1version == -1:
        player1 = User('player1', env.state_size, env.action_size)
    else:
        player1_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, env.input_shape, env.action_size,
                                  config.HIDDEN_CNN_LAYERS)

        if player1version > 0:
            player1_network = player1_NN.read(env.name, run_version, player1version)
            player1_NN.model.set_weights(player1_network.get_weights())
        player1 = Agent('player1', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, player1_NN)

    if player2version == -1:
        player2 = User('player2', env.state_size, env.action_size)
    else:
        player2_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, env.input_shape, env.action_size,
                                  config.HIDDEN_CNN_LAYERS)

        if player2version > 0:
            player2_network = player2_NN.read(env.name, run_version, player2version)
            player2_NN.model.set_weights(player2_network.get_weights())
        player2 = Agent('player2', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, player2_NN)

    scores, memory, points, sp_scores = playMatches(player1, player2, EPISODES, logger, turns_until_tau0, None,
                                                    goes_first)

    return (scores, memory, points, sp_scores)


def playMatches(player1, player2, EPISODES, logger, turns_until_tau0, memory=None, goes_first=0):
    env = Game()
    scores = {player1.name: 0, "drawn": 0, player2.name: 0}
    sp_scores = {'sp': 0, "drawn": 0, 'nsp': 0}
    points = {player1.name: [], player2.name: []}
    e = 0
    # for e in range(EPISODES):
    while e < EPISODES:
        logger.info('====================')
        logger.info('EPISODE %d OF %d', e + 1, EPISODES)
        logger.info('====================')

        # print(str(e + 1) + ' ', end='')
        now = datetime.now()
        ct = now.strftime("%H:%M:%S")
        print("###")
        print(ct)
        print("\nEpisode", e+1)
        state = env.reset()

        done = 0
        turn = 0
        player1.mcts = None
        player2.mcts = None

        if goes_first == 0:
            player1Starts = random.randint(0, 1) * 2 - 1
        else:
            player1Starts = goes_first

        if player1Starts == 1:
            players = {1: {"agent": player1, "name": player1.name}
                , 0: {"agent": player2, "name": player2.name}
                       }
            logger.info(player1.name + ' plays as white')
            print(player1.name, ' plays as white')
        else:
            players = {1: {"agent": player2, "name": player2.name}
                , 0: {"agent": player1, "name": player1.name}
                       }
            logger.info(player2.name + ' plays as white')
            print(player2.name, ' plays as white')
            logger.info('--------------')

        env.gameState.render(logger)

        while done == 0:
            terminated = 0
            turn = turn + 1
            # print(turn,end=' ')
            progress(int(100*turn/MAX_TURNS))
            #### Run the MCTS algo and return an action
            if turn < turns_until_tau0:
                action, pi, MCTS_value, NN_value = players[state.playerTurn]['agent'].act(state, 1)
            else:
                action, pi, MCTS_value, NN_value = players[state.playerTurn]['agent'].act(state, 0)

            if memory != None:
                ####Commit the move to memory
                memory.commit_stmemory(env.identities, state, pi)

            logger.info('action: %s', moves.movelist[action])
            # for r in range(env.grid_shape[0]):
            #     logger.info(['----' if x == 0 else '{0:.2f}'.format(np.round(x,2)) for x in pi[env.grid_shape[1]*r : (env.grid_shape[1]*r + env.grid_shape[1])]])
            logger.info('MCTS perceived value for %s: %f', state.pieces[str(state.playerTurn)], np.round(MCTS_value, 2))
            logger.info('NN perceived value for %s: %f', state.pieces[str(state.playerTurn)], np.round(NN_value, 2))
            logger.info('====================')

            ### Do the action
            state, value, done, _ = env.step(action)
            # the value of the newState from the POV of the new playerTurn i.e. -1 if the previous player played a winning move

            env.gameState.render(logger)
            if turn >= MAX_TURNS:
                terminated = 1
                done = 1
                value = env.gameState.get_value_alpha()[0]
                # print("\n###")
                # print("tree length:", len(player1.mcts.tree))
                # print("value: ", value)
                # bylo env.gameState./ zamiast state./ w 144,147,152,
                if state.get_leader() == 1:
                  winner_print = "white"
                elif state.get_leader() == 0:
                  winner_print = "black"
                else:
                  winner_print = "draw"
                print("winner: ", winner_print)
                print("Board: ", state.board.fen())
                # print("###\n")
            if done == 1:
                if memory != None:
                    #### If the game is finished, assign the values correctly to the game moves
                    for move in memory.stmemory:
                        if terminated:
                            if move['playerTurn'] == state.get_leader():
                                move['value'] = value
                            else:
                                move['value'] = -value
                        else:
                            if move['playerTurn'] == state.playerTurn:
                                move['value'] = value
                            else:
                                move['value'] = -value
                    # training_states = [row['state'] for row in memory.stmemory]
                    # training_targets = {'value_head': np.array([row['value'] for row in memory.stmemory]),
                    #                     'policy_head': np.array([row['AV'] for row in memory.stmemory])}
                    # for i in range(len(memory.stmemory)):
                    #     lg.logger_debug.info("state: " + training_states[i].board.fen())
                    #     lg.logger_debug.info("target value: " + str(training_targets['value_head'][i]))
                    #     lg.logger_debug.info("target policy: " + str(training_targets['policy_head'][i]))

                    memory.commit_ltmemory()

                    # minibatch = random.sample(memory.stmemory, min(config.BATCH_SIZE, len(memory.stmemory)))


                if value == 1:
                    logger.info('%s WINS!', players[state.playerTurn]['name'])
                    scores[players[state.playerTurn]['name']] = scores[players[state.playerTurn]['name']] + 1
                    if state.playerTurn == 1:
                        sp_scores['sp'] = sp_scores['sp'] + 1
                    else:
                        sp_scores['nsp'] = sp_scores['nsp'] + 1

                elif value == -1:
                    logger.info('%s WINS!', players[1 if state.playerTurn == 0 else 0]['name'])
                    scores[players[1 if state.playerTurn == 0 else 0]['name']] = scores[players[1 if state.playerTurn == 0 else 0]['name']] + 1

                    if state.playerTurn == 1:
                        sp_scores['nsp'] = sp_scores['nsp'] + 1
                    else:
                        sp_scores['sp'] = sp_scores['sp'] + 1

                else:
                    logger.info('DRAW...')
                    scores['drawn'] = scores['drawn'] + 1
                    sp_scores['drawn'] = sp_scores['drawn'] + 1

                pts = state.score_alpha if terminated else state.score
                points[players[state.playerTurn]['name']].append(pts[0])
                points[players[1 if state.playerTurn == 0 else 0]['name']].append(-pts[1])
                print(points)
                logger.info("#####")
                logger.info(points)
                logger.info("#####")
            elif done == -1:
                e = e-1
                logger.info("BOARD NOT VALID, REPEATING GAME")
                print(state.board.fen())
                print(state.board.status())
                print("REPEATING GAME...")
        e = e+1


    return (scores, memory, points, sp_scores)
