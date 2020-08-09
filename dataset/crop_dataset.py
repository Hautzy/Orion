from pickle import load

import config as c
from torch.utils.data import Dataset

from preprocessing.sample import Sample
from utils.logger import Logger


class CropDataset(Dataset):
    def __init__(self, name, indices, total_crop_metas):
        self.indices = indices
        self.total_crop_metas = total_crop_metas
        self.sample_count = len(indices)
        self.logger = Logger(f'dataset-{name}')

    def __getitem__(self, index):
        ind = self.indices[index]
        sample_str = self.total_crop_metas[ind]
        return Sample.load(sample_str)

    def __len__(self):
        return self.sample_count