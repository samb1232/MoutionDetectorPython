from time import time

import numpy as np


class FpsCounter:
    def __init__(self, counting_period: int = 1.0):
        self.fps_counter = 0
        self.start_time = time()
        self.fps = 0
        self.counting_period = counting_period

    def update(self) -> None:
        self.fps_counter += 1
        cur_time = time()
        time_diff = cur_time - self.start_time
        if time_diff > self.counting_period:
            self.fps = self.fps_counter / np.round(time_diff)
            self.start_time = time()
            self.fps_counter = 0

    def get_fps(self) -> int:
        return int(self.fps)
