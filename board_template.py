import cv2
import numpy as np



class GridTemplate:

    def __init__(self, max_height, max_width):
        self.height = max_height
        self.width = max_width
        self.grid = self.make_grid()
        self.rows = self.get_rows()

    def make_grid(self):
        img = np.zeros((self.height,self.width))
        step_v = int(self.height/8)
        step_h = int(self.width/8)
        hor = [i for i in range(0,self.height+1,step_v)] # interwały na osi pionowej - pozycje rzędów
        ver = [i for i in range(0,self.width+1,step_h)] # interwały na osi poziomej - pozycje kolumn

        for i in range(9):
            cv2.line(img, (0,hor[i]),(self.width,hor[i]),255,2,cv2.LINE_AA) # rysuje poziome linie
        for i in range(9):
            cv2.line(img, (ver[i],0),(ver[i],self.height),255,2,cv2.LINE_AA)
        # cv2.imshow('sa', img)
        # cv2.waitKey(0)
        return img

    def get_rows(self):
        img = np.zeros((self.height, self.width))
        step_v = int(self.height / 8)
        step_h = int(self.width / 8)
        hor = [i for i in range(0, self.height + 1, step_v)]  # interwały na osi pionowej - pozycje rzędów
        ver = [i for i in range(0, self.width + 1, step_h)]  # interwały na osi poziomej - pozycje kolumn
        rows = []
        for y in hor:
            row = []
            for x in ver:
                pos = [x, y]
                row.append(pos)
            rows.append(row)
        return rows
