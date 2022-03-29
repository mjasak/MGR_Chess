import cv2
import numpy as np


class Square:

    def __init__(self,c1,c2,c3,c4,position,figure=''):
        # Corners
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.position = position
        self.contour = np.array([c1,c2,c4,c3], dtype="int32")
        self.cords = [c1,c2,c3,c4]
        # self.center = self.get_center()
        self.cx, self.cy = self.get_center()
        self.figure = figure
        self.figure_decoded = ''

    def get_center(self):

        # Center of square
        M = cv2.moments(self.contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx = 1
            cy = 1
        return cx,cy

    def draw(self, image, color, thickness=1):
        # Formattign npArray of corners for drawContours
        ctr = np.array(self.contour).reshape((-1, 1, 2)).astype(np.int32)
        cv2.drawContours(image, [ctr], 0, color, 3)
        cv2.putText(image, self.position, (self.cx, self.cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
