import os
import uuid
from pickle import HIGHEST_PROTOCOL, dump

import config as c

from PIL import Image


class Sample:
    def __init__(self, cropped_image, crop_target, crop_map, crop_size, crop_center):
        self.uuid = uuid.uuid4()
        self.cropped_image = cropped_image
        self.crop_target = crop_target
        self.crop_map = crop_map
        self.meta = Meta(crop_size, crop_center)

    def save(self, folder):
        Image.fromarray(self.cropped_image).save(os.path.join(folder, f'{self.uuid}{c.FILE_SAMPLE_CROPPED_IMAGE}'))
        Image.fromarray(self.crop_target).save(os.path.join(folder, f'{self.uuid}{c.FILE_PATH_SAMPLE_CROP_TARGET}'))
        Image.fromarray(self.crop_map).save(os.path.join(folder, f'{self.uuid}{c.FILE_PATH_SAMPLE_CROP_MAP}'))
        with open(os.path.join(folder, f'{self.uuid}{c.FILE_PATH_SAMPLE_META}'), 'wb') as f:
            dump(self.meta, f, protocol=HIGHEST_PROTOCOL)


class Meta:
    def __init__(self, crop_size, crop_center):
        self.crop_size = crop_size
        self.crop_center = crop_center
