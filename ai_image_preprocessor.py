from skimage.transform import resize
import numpy as np

def downsize(img_arr):
    img_arr = resize(img_arr, (84, 84), anti_aliasing=True)
    return img_arr # [:,:84]

def rgb2gray(img_arr):
    return np.true_divide(np.sum(img_arr, axis=-1), 3).astype(np.uint8)

def preprocess(img_arr):
    img_arr = rgb2gray(img_arr)
    downsized = downsize(img_arr)
    return downsized
