import os
import shutil

from utils.logger import Logger

# folders
FOLDER_PATH_MAIN_OUT = '../out'
FOLDER_PATH_PREPROCESSING = FOLDER_PATH_MAIN_OUT + os.sep + 'preprocessing'

FOLDER_PATH_MAIN_IN = 'in'
FOLDER_PATH_RAW_DATA = FOLDER_PATH_MAIN_IN + os.sep + 'raw_data'
# files
FILE_PART_IMAGE_CROP_META_CORSS_REFERENCE = '_crop_metas.csv'
FILE_TOTAL_IMAGE_CROP_META_CROSS_REFERENCE = FOLDER_PATH_PREPROCESSING + os.sep + 'total_crop_metas.csv'
FILE_PART_PART_IMAGE_MEAN = '_part_mean.npz'
FILE_TOTAL_IMAGE_MEAN = FOLDER_PATH_PREPROCESSING + os.sep + 'total_image_mean.npz'

# image constraints
MIN_SAMPLE_WIDTH = 70
MIN_SAMPLE_HEIGHT = 70
MAX_SAMPLE_WIDTH = 100
MAX_SAMPLE_HEIGHT = 100

MIN_CROP_SIZE = 5
MAX_CROP_SIZE = 21

MIN_CROP_TARGET_PIXEL_PADDING = 20

# sample generation
SAMPLE_ROTATION_ANGLES = [0]
RANDOM_CROPS_PER_IMAGE = 2

# other
PADDING_VALUE = 0
MIN_PIXEL_VALUE = 1
MAX_PIXEL_VALUE = 255

logger = Logger('CONFIG')


def create_folders():
    logger.log('creating folder structure')
    create_folder(FOLDER_PATH_MAIN_OUT)
    create_folder(FOLDER_PATH_PREPROCESSING)


def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
        logger.log(f'created folder {path}')


def clear_folders():
    logger.log('clearing folders')
    if os.path.exists(FOLDER_PATH_MAIN_OUT):
        shutil.rmtree(FOLDER_PATH_MAIN_OUT, ignore_errors=True)


def get_all_files_in_folder(folder_path, filter_method=None):
    paths = list()
    for subdir, dirs, files in os.walk(folder_path):
        for file in files:
            if filter_method is not None:
                if filter_method(file):
                    paths.append(os.path.join(subdir, file))
            else:
                paths.append(os.path.join(subdir, file))
    return paths
