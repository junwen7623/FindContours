import numpy as np
from suzuki2 import Suzuki2


if __name__ == '__main__':
    # img = np.array([[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
                    # [1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1],
                    # [1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1],
                    # [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]])
                    
    img = np.array([[1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 0, 0, 1, 0, 0, 1, 0, 1],
                    [1, 0, 0, 1, 0, 0, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0]])

    su = Suzuki2(img)
    su.exec()
    np.savetxt('./outputs/pad.txt', su.pad, fmt='%2d')

    for e in su.contour_list:
        print(e)
