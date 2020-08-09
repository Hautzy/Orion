from pickle import dump, HIGHEST_PROTOCOL

import config as c

class PixelNormalizer:
    def __init__(self, total_mean):
        self.total_mean = 0.0