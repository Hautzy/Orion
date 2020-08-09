import os
import uuid
from pickle import HIGHEST_PROTOCOL, dump, load

import config as c

from PIL import Image


class Sample:

    def __init__(self, cropped_image, crop_target, crop_map, meta, uuidx=None):
        if uuidx is not None:
            self.uuid = uuidx
        else:
            self.uuid = uuid.uuid4()
        self.cropped_image = cropped_image
        self.crop_target = crop_target
        self.crop_map = crop_map
        self.meta = meta

    def save(self, folder):
        Image.fromarray(self.cropped_image).save(self.s_create_path(folder, c.FILE_SAMPLE_CROPPED_IMAGE))
        Image.fromarray(self.crop_target).save(self.s_create_path(folder, c.FILE_PATH_SAMPLE_CROP_TARGET))
        Image.fromarray(self.crop_map).save(self.s_create_path(folder, c.FILE_PATH_SAMPLE_CROP_MAP))
        with open(self.s_create_path(folder, c.FILE_PATH_SAMPLE_META), 'wb') as f:
            dump(self.meta, f, protocol=HIGHEST_PROTOCOL)

    def s_create_path(self, folder, postfix):
        return Sample.create_path(self.uuid, folder, postfix)

    @staticmethod
    def create_path(uuid, folder, postfix):
        return os.path.join(folder, f'{uuid}{postfix}')

    @staticmethod
    def load(folder, uuid):
        cropped_image = Image.open(Sample.create_path(uuid, folder, c.FILE_SAMPLE_CROPPED_IMAGE))
        crop_target = Image.open(Sample.create_path(uuid, folder, c.FILE_PATH_SAMPLE_CROP_TARGET))
        crop_map = Image.open(Sample.create_path(uuid, folder, c.FILE_PATH_SAMPLE_CROP_MAP))
        with open(Sample.create_path(uuid, folder, c.FILE_PATH_SAMPLE_META), 'rb') as f:
            meta = load(f)
        sample = Sample(cropped_image.numpy(), crop_target.numpy(), crop_map.numpy(), meta, uuid)
        return sample

class Meta:
    def __init__(self, crop_size, crop_center):
        self.crop_size = crop_size
        self.crop_center = crop_center
