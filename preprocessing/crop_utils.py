import config as c
import numpy as np
import random


def get_random_crop_size_and_center(image_height, image_width):
    valid_crop = False
    trys = 0
    while not valid_crop and trys < 10:
        crop_width = random.randrange(c.MIN_CROP_SIZE, c.MAX_CROP_SIZE + 1, 2)
        crop_height = random.randrange(c.MIN_CROP_SIZE, c.MAX_CROP_SIZE + 1, 2)

        start_x = c.MIN_CROP_TARGET_PIXEL_PADDING + int(crop_width / 2) + 1
        start_y = c.MIN_CROP_TARGET_PIXEL_PADDING + int(crop_height / 2) + 1
        end_x = image_width - c.MIN_CROP_TARGET_PIXEL_PADDING - int(crop_width / 2) - 1
        end_y = image_height - c.MIN_CROP_TARGET_PIXEL_PADDING - int(crop_height / 2) - 1
        valid_crop = end_y - start_y > 0 and end_x - start_x > 0
        trys += 1

    if not valid_crop:
        return None, None
    center_x = random.randrange(start_x, end_x)
    center_y = random.randrange(start_y, end_y)
    return (crop_height, crop_width), (center_y, center_x)

# calculate start and end x/y coordinates for crop sub-image based on crop_size and crop_center
def calculate_crop_target_coordinates(crop_size, crop_center):
    st_x = int(crop_center[1] - (crop_size[1] - 1) / 2)
    en_x = int(crop_center[1] + (crop_size[1] - 1) / 2) + 1
    st_y = int(crop_center[0] - (crop_size[0] - 1) / 2)
    en_y = int(crop_center[0] + (crop_size[0] - 1) / 2) + 1
    return (st_x, st_y), (en_x, en_y)
