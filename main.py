import numpy as np
import cv2
import chess
import chess.svg
import Chessboard
from Chessboard import Chessboard
from Squares import Square
import os
import sys
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QApplication, QWidget


ROOT_DIR = os.getcwd()
sys.path.append(os.path.join(ROOT_DIR, 'ssd_keras'))
Pts = []


class MainWindow(QWidget):
    def __init__(self,board):
        super().__init__()

        self.setGeometry(100, 100, 520, 520)

        self.widgetSvg = QSvgWidget(parent=self)
        self.widgetSvg.setGeometry(10, 10, 500, 500)

        self.chessboard = board

        self.chessboardSvg = chess.svg.board(self.chessboard).encode("UTF-8")
        self.widgetSvg.load(self.chessboardSvg)


def initialize(color):
    cap = cv2.VideoCapture(0)
    chessb = Chessboard()
    return cap, chessb, color


def read_cam(cap):
    ret, frame = cap.read()
    # frame = cv2.resize(frame, (480,360))
    cv2.imshow('frame',frame)
    cv2.waitKey(0)
    frame = frame[20:460,:]
    return frame


def main():
    # impath = r'C:\Users\jasak\MGR\chessCV\img2.jpg'
    # image = cv2.imread(impath)
    cap = cv2.VideoCapture(0)
    for i in range(10):
        ret, img = cap.read()
        # cv2.imshow('webcam', img)
        # k = cv2.waitKey(10)
        # if k==27:
        #     break;

    ret,image = cap.read()
    ret, image = cap.read()
    image = image[80:440,100:540]
    # cv2.imshow('winname',image)\
    # image = cv2.resize(image,(300,300))
    #
    # impath = r'C:\Users\jasak\MGR\chessCV\0034.jpg'
    # image = cv2.imread(impath)

    orig = image.copy()
    a = Chessboard()
    a.init_board(image)
    cv2.imshow("image to detection", orig)
    cv2.waitKey(0)
    a.warp_grid(orig,a.make_rectangle(orig))
    a.visualizeBoxes(a.detect(orig.copy()),orig.copy())
    ret = a.read_board(orig.copy(), 1)

    app = QApplication([])
    window = MainWindow(a.board)
    window.show()
    app.exec()

    print(a.board.fen())
    print("Valid: ", a.board.is_valid())
    print("Status: ",a.board.status())
    if ret:
        print(a.make_move())


if __name__ == '__main__':
    main()