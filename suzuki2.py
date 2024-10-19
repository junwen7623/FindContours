# Created by Junwen Deng on 2024-10-19.
# Copyright © 2024 Junwen Deng. All rights reserved.
# The python language implementation of Algorithm 2 in the paper.
# Suzuki, S. (1985). Topological structural analysis of digitized binary images by border following. Computer vision, graphics, and image processing, 30(1), 32-46.
# If you find a bug, please let me know. Email: junwen7623@163.com


import numpy as np


class Suzuki2:
    def __init__(self, img):
        assert (img.shape[0] > 1) and (img.shape[1] > 1)
        self.nbd = 1
        self.lnbd = 0
        self.img = img
        self.pad = np.pad(img, ((1, 1), (1, 1)), 'constant', constant_values=0)
        self.ij = (1, 1)  # scan from (1, 1)
        self.i2j2 = None
        self.i1j1 = None
        self.i3j3 = None
        self.is_0_pixel_examined = False
        self.i4j4 = None
        self.neighbours = 8  # num of neighbours. 4 or 8.
        self.contour_list = []
        self.contour = []

    def next_point(self, point):
        i, j = point
        rows, cols = self.img.shape
        if (i == rows) and (j == cols):  # scan to end of self.img, return (-1, -1)
            return -1, -1
        if j < cols:  # move right
            j += 1
        elif j == cols:  # i < rows default here. move to start of new line
            i += 1
            j = 1
            self.lnbd = 0
        return i, j

    def step1(self):
        while True:
            i, j = self.ij
            if (self.pad[i, j] == 1) and (self.pad[i, j - 1] == 0) and (self.lnbd <= 0):
                    self.nbd = 2
                    self.i2j2 = (i, j-1)
                    return False  # go to step 2
            elif (self.pad[i, j] >= 1) and (self.pad[i, j+1] == 0):
                self.nbd = 2
                self.i2j2 = (i, j+1)
                if self.pad[i, j] > 1:
                    self.lnbd = self.pad[i, j]
                return True  # go to step 4
            else :
                if self.pad[i, j] == 0:
                    y, x = self.next_point((i, j))
                    if (y == -1) and (x == -1):  # scan to end of self.img, not change self.ij
                        return True  # go to step 4
                    else:
                        self.ij = (y, x)
                        continue  # skip zero point
                else:
                    return True  # go to step 4

    def step2(self):
        pass

    def n4(self, point):
        i, j = point
        points = [(i, j + 1), (i - 1, j), (i, j - 1), (i + 1, j)]  # 4 neighbours coordinates (anti-clockwise)
        values = [self.pad[i, j] for (i, j) in points]
        return points, values

    def n8(self, point):
        i, j = point
        points = [(i, j+1), (i-1, j+1), (i-1, j), (i-1, j-1), (i, j-1), (i+1, j-1), (i+1, j), (i+1, j+1)]
        values = [self.pad[i, j] for (i, j) in points]
        return points, values

    def step3_1(self):
        if self.neighbours == 4:
            points, values = self.n4(self.ij)
        elif self.neighbours == 8:
            points, values = self.n8(self.ij)
        else:
            raise ValueError('invalid neighbours value.')
        points = points[::-1]
        values = values[::-1]
        i = points.index(self.i2j2)
        points = points[i:] + points[:i]
        values = values[i:] + values[:i]
        indices = [i for i, v in enumerate(values) if v != 0]
        if len(indices):  # not empty
            self.i1j1 = points[indices[0]]
            return False  # go to step 3.2
        else:  # empty
            i, j = self.ij
            self.pad[i, j] = -self.nbd
            self.contour.append((i, j))
            self.contour_list.append(self.contour)
            self.contour = []
            return True  # go to step 4

    def step3_2(self):
        self.i2j2 = self.i1j1
        self.i3j3 = self.ij

    def step3_3(self):
        if self.neighbours == 4:
            points, values = self.n4(self.i3j3)
            i = points.index(self.i2j2)
            s = 0 if i == 3 else i+1
        elif self.neighbours == 8:
            points, values = self.n8(self.i3j3)
            i = points.index(self.i2j2)
            s = 0 if i == 7 else i+1
        else:
            raise ValueError('invalid neighbours value.')
        points = points[s:] + points[:s]
        values = values[s:] + values[:s]
        i3j3_1 = (self.i3j3[0], self.i3j3[1]+1)
        self.is_0_pixel_examined = False
        for p, v in zip(points, values):
            if (v == 0) and (p == i3j3_1):
                self.is_0_pixel_examined = True
            if v: # not zero
                self.i4j4 = p
                break
                
    def step3_4(self):
        i, j = self.i3j3
        if self.is_0_pixel_examined:
            self.pad[i, j] = -self.nbd
        elif (not self.is_0_pixel_examined) and (self.pad[i, j] == 1):
            self.pad[i, j] = self.nbd
        else:
            pass

    def step3_5(self):
        i, j = self.i3j3
        self.contour.append((i, j))
        if (self.i4j4 == self.ij) and (self.i3j3 == self.i1j1):
            self.contour_list.append(self.contour)
            self.contour = []
            return True  # go to step 4
        else:
            self.i2j2 = self.i3j3
            self.i3j3 = self.i4j4
            return False # go to step 3.3

    def step4(self):
        i, j = self.ij
        if self.pad[i, j] != 1:
            self.lnbd = self.pad[i, j]
        rows, cols = self.img.shape
        if (i == rows) and (j == cols):
            return True  # end of algorithm 2
        else:
            self.ij = self.next_point((i, j))
            return False  # go to step 1

    def exec(self):
        while True:
            if not self.step1():
                self.step2()
                if not self.step3_1():
                    self.step3_2()
                    while True:
                        self.step3_3()
                        self.step3_4()
                        if self.step3_5():
                            break
            if self.step4():
                break
