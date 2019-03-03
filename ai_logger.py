from IPython import display 
from collections import deque

class Logger:
    
    def __init__(self, n_rows, header=None):
        self.log_queue = deque(maxlen=n_rows)
        self.header = header
    
    def add(self, text):
        self.log_queue.append(text)
    
    def log(self):
        display.clear_output(wait=True)
        if self.header != None:
            print(self.header)
        print("\n".join(self.log_queue), flush=True)
