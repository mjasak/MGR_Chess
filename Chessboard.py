import numpy as np
import chess
import os
import sys

ROOT_DIR = os.getcwd()
sys.path.append(os.path.join(ROOT_DIR, 'ssd_keras'))

import tensorflow as tf
from keras import backend as K
from matplotlib import pyplot as plt
from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras.optimizers import Adam
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from board_template import GridTemplate
from chess import Board
import cv2
from numpy.linalg import inv
from numpy.linalg import norm
from Squares import Square
from stockfish import Stockfish

Pts = []


class Chessboard:

    def __init__(self):
        self.debug = False
        self.Squares = {}
        self.board = Board().reset()
        self.img_width = 440
        self.img_height = 360
        self.img_channels = 3
        self.n_classes = 13
        self.classes = ['background', 'black-bishop', 'black-king', 'black-knight', 'black-pawn', 'black-queen',
                        'black-rook',
                        'white-bishop', 'white-king', 'white-knight', 'white-pawn', 'white-queen', 'white-rook']
        self.path_to_model = r'C:\Users\jasak\MGR\chessCV\ssd300_training_main.h5'
        self.model = self.load_model()
        self.grid_path = r'C:\Users\jasak\MGR\chessCV\grid_template.jpg'
        self.stockfish_path = r'C://Users//jasak//MGR//Stockfish//stockfish_14.1_win_x64_popcnt'
        self.stockfish = Stockfish(self.stockfish_path)
        self.stockfish.set_depth(12)

    def load_model(self):
        K.clear_session()

        model = ssd_300(image_size=(300, 300, 3),
                        n_classes=self.n_classes,
                        mode='inference',
                        l2_regularization=0.0005,
                        scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05],
                        aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                 [1.0, 2.0, 0.5],
                                                 [1.0, 2.0, 0.5]],
                        two_boxes_for_ar1=True,
                        steps=[8, 16, 32, 64, 100, 300],
                        offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                        clip_boxes=False,
                        variances=[0.1, 0.1, 0.2, 0.2],
                        normalize_coords=True,
                        subtract_mean=[123, 117, 104],
                        swap_channels=[2, 1, 0],
                        confidence_thresh=0.5,
                        iou_threshold=0.45,
                        top_k=200,
                        nms_max_output_size=400)
        model.load_weights(self.path_to_model, by_name=True)
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
        model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
        if self.debug:
            model.summary()
        return model

    def detect(self, image):
        images = [image]
        input_images = [image]
        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
        input_images[0] = cv2.resize(input_images[0], (300, 300))
        input_images = np.array(input_images)
        y_pred = self.model.predict(input_images)
        confidence_threshold = 0.5
        y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold] for k in range(y_pred.shape[0])]
        return y_pred_thresh

    def visualizeBoxes(self, y_pred_thresh, recimg):

        np.set_printoptions(precision=2, suppress=True, linewidth=90)

        orig_images = [cv2.resize(recimg, (640, 480))]
        img_width = 300
        img_height = 300
        imag = orig_images[0].copy()
        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

        for box in y_pred_thresh[0]:
            # Transform the predicted bounding boxes for the 416x416 image to the original image dimensions.
            xmin = box[2] * orig_images[0].shape[1] / img_width
            ymin = box[3] * orig_images[0].shape[0] / img_height
            xmax = box[4] * orig_images[0].shape[1] / img_width
            ymax = box[5] * orig_images[0].shape[0] / img_height
            color = colors[int(box[0])]
            label = '{}: {:.2f}'.format(self.classes[int(box[0])], box[1])
            pnt1 = (int(xmin), int(ymin))
            pnt2 = (int(xmax), int(ymax))
            pnt1b = (int(xmin), int(ymin - 5))
            cv2.rectangle(img=imag, pt1=pnt1, pt2=pnt2, color=(255, 0, 0), thickness=2)
            cv2.putText(
                img=imag,
                text=label,
                org=pnt1b,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 0, 0),
                thickness=2
            )
        cv2.imshow('screen', imag)
        cv2.waitKey(0)

    def get_rows(self, image, M, rows):

        rows_warped = []
        for row in rows:
            row_array = np.array(row, dtype="float32")
            row_array = np.reshape(row_array, (9, 1, 2))
            warped_row = cv2.perspectiveTransform(row_array, inv(M))
            warped = []
            for point in warped_row:
                center = list([int(l) for l in point[0]])
                warped.append(center)
            rows_warped.append(warped)

        return rows_warped

    def get_m(self, image, rectangle):
        rectangle = np.array(rectangle, dtype='float32')
        (a, b, c, d) = rectangle

        width1 = norm(c - d)
        width2 = norm(b - a)
        max_width = max(int(width1), int(width2))

        height1 = norm(b - c)
        height2 = norm(a - d)
        max_height = max(int(height1), int(height2))
        print(max_width, max_height)
        grid = GridTemplate(max_height, max_width).grid
        rows = GridTemplate(max_height, max_width).rows

        vertices = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype='float32')
        M = cv2.getPerspectiveTransform(rectangle, vertices)
        return M, rows

    def warp_grid(self, image, rectangle):
        rectangle = np.array(rectangle, dtype='float32')
        (a, b, c, d) = rectangle

        width1 = norm(c - d)
        width2 = norm(b - a)
        max_width = max(int(width1), int(width2))

        height1 = norm(b - c)
        height2 = norm(a - d)
        max_height = max(int(height1), int(height2))
        print(max_width, max_height)
        grid = GridTemplate(max_height, max_width).grid

        vertices = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype='float32')
        M = cv2.getPerspectiveTransform(rectangle, vertices)
        # out = cv2.warpPerspective(src=image, M=M, dsize=(max_width, max_height))
        # cv2.imshow('out',out)
        # cv2.waitKey(0)
        print(image.shape)
        w,h,d = image.shape
        warped_grid = cv2.warpPerspective(src=grid, M=inv(M), dsize=(h,w))
        cv2.imshow('warped_grid', warped_grid)
        cv2.waitKey(0)

        # poly = np.array([[0, 0], [max_width, max_height], [max_width, 0], [0, max_height]], dtype="float32")
        # polygon = np.reshape(poly, (poly.shape[0], 1, 2))
        # warped = cv2.perspectiveTransform(polygon, inv(M))

    def make_rectangle(self, image):

        def get_pos(event, x, y, flags, param):
            global Pts
            if event == cv2.EVENT_LBUTTONDBLCLK:
                Pts.append([x, y])
                cv2.circle(image, tuple(Pts[-1]), 3, (255, 0, 0), 1)

        cv2.namedWindow('image')
        cv2.setMouseCallback('image',get_pos)

        while len(Pts) < 4:
            cv2.imshow('image', image)
            ke = cv2.waitKey(20) & 0xFF
            if ke == 27:
                break
        # print(Pts)
        # return [[49, 294], [387, 296], [348, 42], [90, 33]]
        return Pts

    def init_squares(self, image, cols):

        letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        numbers = ['1', '2', '3', '4', '5', '6', '7', '8']

        # initialize squares
        for r in range(0, 8):
            for c in range(0, 8):
                c1 = cols[r][c]
                c2 = cols[r][c + 1]
                c3 = cols[r + 1][c]
                c4 = cols[r + 1][c + 1]

                position = letters[c] + numbers[r]
                newSquare = Square(c1, c2, c3, c4, position)
                newSquare.draw(image, (255, 0, 0), 2)
                self.Squares[str(position)] = newSquare
        cv2.imshow('squares', image)
        cv2.waitKey(0)

    def assign(self, y_pred_thresh, orig_image):
        img_width = 300
        img_height = 300
        orig_images = [orig_image]
        if len(y_pred_thresh[0]) > 0:
            for box in y_pred_thresh[0]:
                xmin = box[2] * orig_images[0].shape[1] / img_width
                ymin = box[3] * orig_images[0].shape[0] / img_height
                xmax = box[4] * orig_images[0].shape[1] / img_width
                ymax = box[5] * orig_images[0].shape[0] / img_height
                label = self.classes[int(box[0])]
                mindist = 600 * 600
                index = 65
                minkey = ''
                xav = int((xmin + xmax) / 2)
                yav = int(ymin + 0.9 * (ymax - ymin))
                for key, value in self.Squares.items():

                    # y = value.c3[1]
                    # x = int((value.c1[0] + value.c2[0])/2)
                    y = value.cy
                    x = value.cx
                    dist = abs((xav - x) ** 2 + (yav - y) ** 2)

                    if dist < mindist:
                        mindist = dist
                        minkey = key
                cv2.circle(orig_image, (xav, int(yav)), 5, (255, 0, 0), 3)
                self.Squares[minkey].figure = label

    def decode_figures(self):

        for key, value in self.Squares.items():
            if value.figure == 'black-pawn':
                value.figure_decoded = 'p'
            elif value.figure == 'black-knight':
                value.figure_decoded = 'n'
            elif value.figure == 'black-bishop':
                value.figure_decoded = 'b'
            elif value.figure == 'black-queen':
                value.figure_decoded = 'q'
            elif value.figure == 'black-king':
                value.figure_decoded = 'k'
            elif value.figure == 'black-rook':
                value.figure_decoded = 'r'
            elif value.figure == 'white-pawn':
                value.figure_decoded = 'P'
            elif value.figure == 'white-knight':
                value.figure_decoded = 'N'
            elif value.figure == 'white-bishop':
                value.figure_decoded = 'B'
            elif value.figure == 'white-queen':
                value.figure_decoded = 'Q'
            elif value.figure == 'white-king':
                value.figure_decoded = 'K'
            elif value.figure == 'white-rook':
                value.figure_decoded = 'R'
            else:
                value.figure_decoded = '0'

    def array_to_board(self, helper):
        rows = []
        for s in helper:
            strng = []
            temp = 0
            for ind, char in enumerate(s):
                if char.isdigit():
                    temp += 1
                    if ind == len(s) - 1:
                        strng.append(str(temp))
                else:
                    if temp:
                        strng.append(str(temp))
                        strng.append(char)
                        temp = 0
                    else:
                        strng.append(char)

            rows.append(''.join(strng))

        state_in_string = '/'.join(rows)
        return chess.Board(state_in_string)

    def build_board(self):
        self.decode_figures()
        letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        numbers = ['1', '2', '3', '4', '5', '6', '7', '8']
        board = np.zeros((8, 8), dtype="str")
        for i in range(8):
            for j in range(8):
                position = letters[j] + numbers[7 - i]
                board[i][j] = self.Squares[position].figure_decoded
        chessboard = self.array_to_board(board)
        return chessboard

    def init_board(self,img):
        M, rows = self.get_m(img, self.make_rectangle(img))
        rows_perspective = self.get_rows(img, M, rows)
        self.init_squares(img, rows_perspective)

    def read_board(self,img,turn):
        # a.warp_grid(img, a.make_rectangle(img))
        # M, rows = self.get_m(img, self.make_rectangle(img))
        # rows_perspective = self.get_rows(img, M, rows)
        preds = self.detect(img)
        # self.init_squares(img, rows_perspective)
        self.assign(preds, img)
        self.board = self.build_board()
        self.board.turn = turn
        print(self.board)
        if self.board.is_valid():
            return True
        else:
            return False

    def make_move(self):
        self.stockfish.set_fen_position(self.board.fen())
        return self.stockfish.get_best_move()
