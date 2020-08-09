from pickle import load

import config as c
from torch.utils.data import Dataset

from preprocessing.sample import Sample
from utils.logger import Logger


class CropDataset(Dataset):
    def __init__(self, name, sample_uuids, folder):
        self.sample_uuids = sample_uuids
        self.folder = folder
        self.sample_count = len(sample_uuids)
        self.logger = Logger(f'dataset-{name}')

    def __getitem__(self, index):
        uuid = self.sample_uuids[index]
        return Sample.load(self.folder, uuid)

    def __len__(self):
        return self.sample_count