import chess
import numpy as np
# np.set_printoptions(suppress=True)
import moves
from game import Game, GameState
# import itertools
from chess import Board
import itertools
from keras.utils.vis_utils import plot_model
from model import Residual_CNN
import config
env=Game()
model = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, env.input_shape, env.action_size,
                          config.HIDDEN_CNN_LAYERS)

# plot_model(model.model, to_file='plot_model.png', show_shapes=True)
model.model.compile()
model.model.summary()