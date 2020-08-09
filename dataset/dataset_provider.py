import os
import config as c
import numpy as np

from dataset.crop_dataset import CropDataset
from preprocessing.pixel_normalizer import PixelNormalizer
from utils.logger import Logger


class DataSetProvider:
    def __init__(self, shuffle=False, train_ratio=0.8, test_ratio=0.1, validation_ratio=0.1):
        self.shuffle = shuffle
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.validation_ratio = validation_ratio
        self.logger = Logger(f'DATASETPROVIDER')

    def determine_datasets_sample_distribution(self):
        with open(c.FILE_TOTAL_IMAGE_CROP_META_CROSS_REFERENCE, 'r') as f:
            sample_count = len(f.readlines())
            self.logger.log(f'{sample_count} samples were found')

            test_set_size = int(np.ceil(sample_count * self.test_ratio))
            validation_set_size = int(np.ceil(sample_count * self.validation_ratio))

            shuffled_indices = np.random.permutation(sample_count) if self.shuffle else np.arange(sample_count)
            test_set_indices = shuffled_indices[:test_set_size]
            validation_set_indices = shuffled_indices[test_set_size:(test_set_size+validation_set_size)]
            train_set_indices = shuffled_indices[test_set_size+validation_set_size:]

            return train_set_indices, test_set_indices, validation_set_indices
        return None

    def create_datasets(self):
        # TODO: apply new changes to dataset determination
        train_set_indices, test_set_indices, validation_set_indices = self.determine_datasets_sample_distribution()

        total_sample_len = len(train_set_indices) + len(test_set_indices) + len(validation_set_indices)

        normalizer = PixelNormalizer(indices=train_set_indices, process_nom=3)
        normalizer.start_fit()
        normalizer.start_transform(total_sample_len)

        #train_set = CropDataset('train', train_set_indices, self.data_folder)
        #test_set = CropDataset('test', test_set_indices, self.data_folder)
        #validation_set = CropDataset('validation', validation_set_indices, self.data_folder)
        #return train_set, test_set, validation_set
