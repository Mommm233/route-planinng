import cv2
import numpy as np

# path = './image/map.png'

class Getmap():
    def __init__(self, path, rate=0.6):
        img = cv2.imread(path)
        h, w, _ = img.shape
        rate = 0.6
        self.map = cv2.resize(img, (int(h * rate), int(w * rate)))[:, :, 0] // 255
        self.map = np.array(self.map)
        self.map[self.map == 1] = 255



