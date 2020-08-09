import config as c
from dataset.dataset_provider import DataSetProvider
from preprocessing.preprocessing_pool import PreprocessingPool

c.clear_folders()
c.create_folders()

prepro_pool = PreprocessingPool(process_nom=1)
prepro_pool.start()

dsp = DataSetProvider(c.FOLDER_PATH_PREPROCESSING)
dsp.create_datasets()