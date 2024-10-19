import numpy as np
import cv2 as cv
from suzuki2 import Suzuki2


if __name__ == '__main__':
    img = cv.imread('./imgs/img5.jpg', cv.IMREAD_GRAYSCALE)
    ret, bin = cv.threshold(img, 128, 255, cv.THRESH_BINARY_INV)
    bin[bin > 128] = 1
    img = bin.astype(int)

    su = Suzuki2(img)
    su.exec()
    np.savetxt('./outputs/pad.txt', su.pad, fmt='%2d')

    for e in su.contour_list:
        print(e)


