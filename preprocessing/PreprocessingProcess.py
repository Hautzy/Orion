from multiprocessing import Process, Queue

from PIL import Image

import config as c
import numpy as np

from preprocessing.Sample import Sample
from preprocessing.crop_utils import calculate_crop_target_coordinates, get_random_crop_size_and_center
from utils.logger import Logger


class PreprocessingProcess(Process):

    def __init__(self, idx, workload):
        super(PreprocessingProcess, self).__init__()
        self.id = idx
        self.workload = workload
        self.logger = Logger(f'PREPRO-{self.id}')

    def run(self):
        skipped_samples = 0
        sample_count = 1
        for i, image_path in enumerate(self.workload):
            image = Image.open(image_path)
            image = self.scale_image(image)
            for j, deg in enumerate(c.SAMPLE_ROTATION_ANGLES):
                image_copy = image.copy()
                rotated_image = self.rotate_image(image_copy, deg)
                image_array = np.array(rotated_image, dtype='uint8')
                for k in range(c.RANDOM_CROPS_PER_IMAGE):
                    sample = self.create_sample(image_array)
                    if sample is None:
                        skipped_samples += 1
                        continue
                    sample.save(c.FOLDER_PATH_PREPROCESSING)
                    if sample_count % 1000 == 0:
                        self.logger.log(f'{sample_count} samples created so far')
                    sample_count += 1
        self.logger.log(f'{sample_count-1} samples created in total, skipped {skipped_samples} samples')

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
        (st_x, st_y), (en_x, en_y) = calculate_crop_target_coordinates(crop_size, crop_center)
        cropped_image = np.copy(image_array)
        crop_target = cropped_image[st_y:en_y, st_x:en_x].copy()
        cropped_image[st_y:en_y, st_x:en_x] = c.MIN_PIXEL_VALUE
        crop_map = np.full(shape=cropped_image.shape, fill_value=c.PADDING_VALUE, dtype='uint8')
        crop_map[st_y:en_y, st_x:en_x] = c.MAX_PIXEL_VALUE

        return Sample(cropped_image, crop_target, crop_map, crop_size, crop_center)
