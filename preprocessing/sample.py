import os
import uuid
import numpy as np
from pickle import HIGHEST_PROTOCOL, dump, load

import torch

import config as c

from PIL import Image

from preprocessing.crop_meta import convert_str_to_crop_meta


class Sample:

    def __init__(self, X, y, map, crop_meta):
        self.X = X
        self.y = y
        self.map = map
        self.crop_meta = crop_meta

    def to(self, device):
        self.X = self.X.to(device)
        self.y = self.y.to(device)
        self.map = self.map.to(device)
        return self

    @staticmethod
    def create_path(uuid, folder, postfix):
        return os.path.join(folder, f'{uuid}{postfix}')

    @staticmethod
    def load(sample_str):
        parts = sample_str.split(';')
        image_uuid = parts[0]
        crop_meta = convert_str_to_crop_meta(';'.join(parts[1:]))

        with open(f'{c.FOLDER_PATH_PREPROCESSING}{os.sep}{image_uuid}.npy', 'rb') as f:
            np_X = np.load(f)
            t_X = torch.from_numpy(np_X)

        (st_x, st_y), (en_x, en_y) = crop_meta.get_coordinates()
        t_y = t_X[st_y:en_y, st_x:en_x].clone().detach()
        t_X[st_y:en_y, st_x:en_x] = c.MIN_PIXEL_VALUE
        t_map = torch.zeros(t_X.shape)
        t_map[st_y:en_y, st_x:en_x] = c.MAP_POS

        return Sample(t_X, t_y, t_map, crop_meta).to(c.device)