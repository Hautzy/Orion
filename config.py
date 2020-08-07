import os
import shutil

from utils.logger import Logger

# folders
FOLDER_PATH_MAIN_OUT = '../out'
FOLDER_PATH_PREPROCESSING = FOLDER_PATH_MAIN_OUT + os.sep + 'proprocessing'

FOLDER_PATH_MAIN_IN = 'in'
FOLDER_PATH_RAW_DATA = FOLDER_PATH_MAIN_IN + os.sep + 'raw_data'
# files
FILE_SAMPLE_CROPPED_IMAGE = '_cropped_images.jpg'
FILE_PATH_SAMPLE_CROP_TARGET = '_crop_target.jpg'
FILE_PATH_SAMPLE_CROP_MAP = '_crop_map.jpg'
FILE_PATH_SAMPLE_META = '_meta.pkl'

# image constraints
MIN_SAMPLE_WIDTH = 70
MIN_SAMPLE_HEIGHT = 70
MAX_SAMPLE_WIDTH = 100
MAX_SAMPLE_HEIGHT = 100

MIN_CROP_SIZE = 5
MAX_CROP_SIZE = 21

MIN_CROP_TARGET_PIXEL_PADDING = 20

# sample generation
SAMPLE_ROTATION_ANGLES = [0, 180]
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


def get_all_files_in_folder(folder_path):
    paths = list()
    for subdir, dirs, files in os.walk(folder_path):
        for file in files:
            paths.append(os.path.join(subdir, file))
    return paths
