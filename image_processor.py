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

    def show (self, img):
        cv2.imshow("Image", img)
        cv2.waitKey(100)
