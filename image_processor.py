import numpy as np
import cv2

class ImageProcessor:

    def __init__(self):
        pass

    def to_grayscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def downsample(self, img, dim):
        return cv2.resize(img, dsize=dim, interpolation=cv2.INTER_AREA)

    def preprocess(self, img, dim):
        return self.to_grayscale(self.downsample(img, dim))

    def scaledDimensions(self, width, height, scale):
        return int(width * scale), int(height * scale)

    def show(self, img):
        img1 = np.array(img[0], dtype=np.uint8)
        for x in range(0, 84):
            for y in range(0, 84):
                img1[x][y] = np.uint8((img[0][x][y] + img[1][x][y] + img[2][x][y] + img[3][x][y]) // 4)
        cv2.imshow("Image", img1)

