import os
import config as c
import numpy as np

from dataset.crop_dataset import CropDataset
from utils.logger import Logger


class DataSetProvider:
    def __init__(self, data_folder, shuffle=False, train_ratio=0.8, test_ratio=0.1, validation_ration=0.1):
        self.data_folder = data_folder
        self.shuffle = shuffle
        self.train_ration = train_ratio
        self.test_ration = test_ratio
        self.validation_ration = validation_ration
        self.logger = Logger(f'DATASETPROVIDER')

    def determine_datasets_sample_distribution(self):
        sample_meta_files = c.get_all_files_in_folder(self.data_folder, lambda f: c.FILE_PATH_SAMPLE_META in f)
        sample_uuids = list()
        for sample_meta_file in sample_meta_files:
            sample_uuid = sample_meta_file.replace(c.FILE_PATH_SAMPLE_META, '').replace(self.data_folder + os.sep, '')
            sample_uuids.append(sample_uuid)
        sample_count = len(sample_uuids)
        self.logger.log(f'{sample_count} samples were found')

        test_set_size = int(np.ceil(sample_count * self.test_ratio))
        validation_set_size = int(np.ceil(sample_count * self.validation_ratio))

        shuffled_indices = np.random.permutation(sample_count) if self.shuffle else np.arange(sample_count)
        test_set_indices = shuffled_indices[:test_set_size]
        validation_set_indices = shuffled_indices[test_set_size:(test_set_size+validation_set_size)]
        train_set_indices = shuffled_indices[test_set_size+validation_set_size:]

        train_set_uuids = [sample_uuids[ind] for ind in train_set_indices]
        test_set_uuids = [sample_uuids[ind] for ind in test_set_indices]
        validation_set_uuids = [sample_uuids[ind] for ind in validation_set_indices]
        return train_set_uuids, test_set_uuids, validation_set_uuids

    def create_datasets(self):
        train_set_uuids, test_set_uuids, validation_set_uuids = self.determine_datasets_sample_distribution()
        train_set = CropDataset('train', train_set_uuids, self.data_folder)
        test_set = CropDataset('test', test_set_uuids, self.data_folder)
        validation_set = CropDataset('validation', validation_set_uuids, self.data_folder)
        return train_set, test_set, validation_set
