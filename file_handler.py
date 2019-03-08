

class FileHandler:


    def __init__(self):
        self.steps = self._load_steps()


    def _load_steps(self):
        step_file = open("step_file.txt","r") #only read here
        line = step_file.read()
        if line:
            return int(line)
        else:
            return 0
        step_file.close()
                

    def write_to_file(self, steps):
        with open('step_file.txt', 'w') as filetowrite:
            filetowrite.write(str(steps))
            filetowrite.close()
        


    def set_steps(self, steps):
        self.steps = steps

    def get_steps(self):
        return self.steps