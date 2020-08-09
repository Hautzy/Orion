import os
import uuid
from multiprocessing import Process

import numpy as np
from PIL import Image

import config as c
from preprocessing.crop_meta import CropMeta
from preprocessing.crop_utils import get_random_crop_size_and_center
from utils.logger import Logger


class PreprocessingProcess(Process):

    def __init__(self, idx, workload):
        super(PreprocessingProcess, self).__init__()
        self.id = idx
        self.workload = workload
        self.workload_length = len(self.workload)
        self.logger = Logger(f'PREPRO-{self.id}')
        self.image_crop_metas = {}

    def run(self):
        skipped_samples = 0
        sample_count = 1
        for i, image_path in enumerate(self.workload):
            image = Image.open(image_path)
            image = self.scale_image(image)
            for j, deg in enumerate(c.SAMPLE_ROTATION_ANGLES):
                image_copy = image.copy()
                rotated_image = self.rotate_image(image_copy, deg)
                image_array = np.array(rotated_image, dtype='float32')
                image_array /= c.MAX_PIXEL_VALUE    # normalize

                image_uuid = uuid.uuid4()
                crop_metas = list()
                self.image_crop_metas[image_uuid] = crop_metas
                for k in range(c.RANDOM_CROPS_PER_IMAGE):
                    crop_meta = self.create_sample(image_array)
                    if crop_meta is None:
                        skipped_samples += 1
                        continue
                    crop_metas.append(crop_meta)
                    if sample_count % 1000 == 0:
                        self.logger.log(f'{sample_count} samples created so far')
                    sample_count += 1
                if len(crop_metas) > 0:
                    with open(f'{c.FOLDER_PATH_PREPROCESSING}{os.sep}{image_uuid}.npy', 'wb') as f:
                        np.save(f, image_array)
        self.logger.log(f'{sample_count-1} samples created in total, skipped {skipped_samples} samples')
        self.write_cross_metas()

    def write_cross_metas(self):
        with open(f'{c.FOLDER_PATH_PREPROCESSING}{os.sep}{self.id}{c.FILE_PART_IMAGE_CROP_META_CORSS_REFERENCE}', 'w') as fh:
            for image_uuid in self.image_crop_metas:
                for crop_meta in self.image_crop_metas[image_uuid]:
                    print(f'{image_uuid};{crop_meta}', file=fh)


    @staticmethod
    def scale_image(img):
        shape = (c.MAX_SAMPLE_WIDTH, c.MAX_SAMPLE_HEIGHT)
        img.thumbnail(shape, Image.ANTIALIAS)
        return img

    @staticmethod
    def rotate_image(img, deg):
        if deg != 0:
            return img.rotate(deg, fillcolor=0)
        return img

    @staticmethod
    def create_sample(image_array):
        height, width = image_array.shape
        crop_size, crop_center = get_random_crop_size_and_center(height, width)
        if crop_size is None or crop_center is None:
            return None
        return CropMeta(crop_size, crop_center)
