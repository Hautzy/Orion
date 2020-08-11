import torch

import config as c
from torch.utils.data import DataLoader

from dataset.dataset_provider import DataSetProvider


def crop_collate_fn(batch_as_list):
    batch_length = len(batch_as_list)

    X = torch.zeros(size=(batch_length, 2, c.MAX_SAMPLE_WIDTH, c.MAX_SAMPLE_HEIGHT))
    y = torch.zeros(size=(batch_length, 1, c.MAX_CROP_SIZE, c.MAX_CROP_SIZE))
    crop_metas = list()

    for i, sample in enumerate(batch_as_list):
        X_shape = sample.X.shape
        X[i, 0, :X_shape[0], :X_shape[1]] = sample.X
        X[i, 1, :X_shape[0], :X_shape[1]] = sample.map
        y_shape = sample.y.shape
        y[i, 0, :y_shape[0], :y_shape[1]] = sample.y
        crop_metas.append(sample.crop_meta)

    return X, y, crop_metas


def create_data_loaders(train_workers=0, test_workers=0, validation_workers=0):
    dataset_provider = DataSetProvider(c.FOLDER_PATH_PREPROCESSING)
    train_set, test_set, validation_set = dataset_provider.create_datasets()

    train_loader = DataLoader(train_set, batch_size=c.TRAIN_BATCH_SIZE, num_workers=train_workers,
                              collate_fn=crop_collate_fn)
    test_loader = DataLoader(test_set, batch_size=c.TEST_BATCH_SIZE, num_workers=test_workers,
                             collate_fn=crop_collate_fn)
    validation_loader = DataLoader(validation_set, batch_size=c.VALIDATION_BATCH_SIZE, num_workers=validation_workers,
                                   collate_fn=crop_collate_fn)

    return train_loader, test_loader, validation_loader
