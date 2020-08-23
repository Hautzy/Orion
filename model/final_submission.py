import os
from pickle import load, dump

import torch

import config as c
import numpy as np
import feature_flag as ff
from preprocessing.crop_meta import CropMeta

from preprocessing.pixel_normalizer import get_latest_total_mean
from preprocessing.sample import Sample


def create_final_predictions(experiment_name):
    with open(c.FINAL_SUBMISSION_RAW_DATA, 'rb') as f:
        test_set = load(f)

    model = torch.load(f'{c.FOLDER_PATH_MAIN_EXPERIMENTS}{os.sep}{experiment_name}{os.sep}best_model.pk')
    crop_sizes = test_set['crop_sizes']
    crop_centers = test_set['crop_centers']
    images = test_set['images']
    model = model.to(c.device)
    if ff.NORMALIZE_DATA_GLOBAL_MEAN:
        total_pixel_mean = get_latest_total_mean()
    outputs = list()

    for ind in range(len(images)):
        crop_size = crop_sizes[ind]
        crop_center = crop_centers[ind]
        image = images[ind]
        image = np.array(image, dtype='float64')

        if ff.SCALE_DATA_0_1:
            image /= c.MAX_PIXEL_VALUE
        if ff.NORMALIZE_DATA_GLOBAL_MEAN:
            image -= total_pixel_mean

        crop_meta = CropMeta(crop_size, crop_center)

        t_X = torch.from_numpy(image)
        (st_x, st_y), (en_x, en_y) = crop_meta.get_coordinates()
        t_y = t_X[st_y:en_y, st_x:en_x].clone().detach()
        t_X[st_y:en_y, st_x:en_x] = c.PADDING_VALUE
        t_map = torch.zeros(t_X.shape)
        t_map[st_y:en_y, st_x:en_x] = c.MAP_POS

        sample = Sample(t_X, t_y, t_map, crop_meta)

        X = torch.zeros(size=(1, 2, c.MAX_SAMPLE_WIDTH, c.MAX_SAMPLE_HEIGHT)).to(c.device)
        y = torch.zeros(size=(1, 1, c.MAX_CROP_SIZE, c.MAX_CROP_SIZE)).to(c.device)
        X_shape = sample.X.shape
        X[0, 0, :X_shape[0], :X_shape[1]] = sample.X
        X[0, 1, :X_shape[0], :X_shape[1]] = sample.map
        y_shape = sample.y.shape
        y[0, 0, :y_shape[0], :y_shape[1]] = sample.y

        output = model(X)
        cpu_output = output.cpu().detach().numpy()
        del output

        if ff.NORMALIZE_DATA_GLOBAL_MEAN:
            cpu_output += total_pixel_mean
        if ff.SCALE_DATA_0_1:
            cpu_output *= c.MAX_PIXEL_VALUE

        result = np.array(cpu_output[0, 0, st_y:en_y, st_x:en_x], dtype='uint8')

        outputs.append(result)  # add crop out result from model
    print(f'Tested {ind + 1} submission samples')
    del model
    with open(f'{c.FINAL_SUBMISSION_FOLDER}{os.sep}submission.pk', 'wb') as f:
        dump(outputs, f)