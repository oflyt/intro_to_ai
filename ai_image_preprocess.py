import numpy as np

def downsize(img_arr):
    return img_arr[::2, ::2]

def rgb2gray(img_arr):
    return np.true_divide(np.sum(img_arr, axis=-1), 3).astype(np.uint8)

def preprocess(img_arr):
    downsized = downsize(img_arr)
    gray = rgb2gray(downsized)
    return gray
