import os

from PIL import Image

import config as c
import numpy as np

from preprocessing.pixel_normalizer import get_latest_total_mean


def inspect_image_by_uuid(image_uuid):
    with open(f'{c.FOLDER_PATH_PREPROCESSING}{os.sep}{image_uuid}.npy', 'rb') as f:
        image = np.load(f)
        image = restore_image_arr(image)
        image.save(f'{c.FOLDER_PATH_INSPECTION}{os.sep}{image_uuid}.jpg')


def restore_image_arr(image):
    image += get_latest_total_mean()
    image *= c.MAX_PIXEL_VALUE
    image = image.astype(dtype='uint8')
    image = Image.fromarray(image)
    return image
