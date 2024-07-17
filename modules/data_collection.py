import numpy as np

class DataCollector:
    def __init__(self):
        self.HEIGHT = 84
        self.WIDTH = 84
        self.CHANNELS = 3

    def collect_data(self):
        # Simulate data collection by generating a random image
        return np.random.randint(0, 256, size=(self.HEIGHT, self.WIDTH, self.CHANNELS), dtype=np.uint8)