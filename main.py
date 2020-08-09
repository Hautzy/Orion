import config as c
from dataset.data_inspector import inspect_image_by_uuid
from dataset.dataset_provider import DataSetProvider
from preprocessing.pixel_normalizer import get_latest_total_mean, read_all_uuids
from preprocessing.preprocessing_pool import PreprocessingPool
'''
c.clear_folders()
c.create_folders()

prepro_pool = PreprocessingPool(process_nom=1)
prepro_pool.start()

dsp = DataSetProvider(c.FOLDER_PATH_PREPROCESSING)
train, test, val = dsp.create_datasets()

print(get_latest_total_mean())
'''

c.create_folders()
for image_uuid in set(read_all_uuids()):
    inspect_image_by_uuid(image_uuid)