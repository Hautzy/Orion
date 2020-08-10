import config as c
from dataset.data_inspector import inspect_image_by_uuid
from dataset.dataset_provider import DataSetProvider
from model.submission import create_submission_predictions, scoring
from model.train import train
from preprocessing.pixel_normalizer import get_latest_total_mean, read_all_uuids
from preprocessing.preprocessing_pool import PreprocessingPool

def create_and_scale_data():
    c.clear_folders()
    c.create_folders()

    prepro_pool = PreprocessingPool(process_nom=2)
    prepro_pool.start()

    dsp = DataSetProvider(c.FOLDER_PATH_PREPROCESSING)
    dsp.scale_datasets(process_nom=2)

def start_training():
    c.create_folders()
    train('ex1')

def start_testing():
    create_submission_predictions('ex1')
    scoring()

#create_and_scale_data()
start_training()
#start_testing()