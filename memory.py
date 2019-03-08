from collections import deque
import random 
import numpy as np


class Memory:

    def __init__(self):
        self.memory             = deque(maxlen=200000)
        self.temp_memory        = [] # TODO: Should probably also be replaced by a deque 
                                     # to improve the readability and also allow the removal
                                     # of the add_to_tmp_memory function


    def add_to_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    # If 6 items are in the temporary memory, we will rewrite the images
    # to different indexes. Images 0, 1, 2, 3 represent the previous state
    # and images 2, 3, 4, 5 represent the current state
    # If there aren't 6 images in the temporary memory the image is simply
    # appended to the array
    def add_to_temp_memory(self, image):
        if len(self.temp_memory) == 6:
            self.temp_memory = [
                self.temp_memory[1],
                self.temp_memory[2],
                self.temp_memory[3],
                self.temp_memory[4],
                self.temp_memory[5],
                image
            ]
        else:
            self.temp_memory.append(image)


    def get_previous_state_from_tmp_memory(self):
        previous_state = [
            self.temp_memory[0],
            self.temp_memory[1],
            self.temp_memory[2],
            self.temp_memory[3],
        ]
        return np.expand_dims(previous_state, axis=0)


    def get_current_state_from_tmp_memory(self):
        current_state = [
            self.temp_memory[2],
            self.temp_memory[3],
            self.temp_memory[4],
            self.temp_memory[5],
        ]
        return np.expand_dims(current_state, axis=0)

    
    def get_batch(self, batch_size=32):
        batch = random.sample(self.memory, batch_size)
        return batch