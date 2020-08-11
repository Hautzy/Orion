import os
from pickle import dump, load

import torch
import dill as pkl
import numpy as np
from PIL import Image

import config as c


# calculate MSE from predicted cropped out sub-image and the correct test sub-image
from preprocessing.crop_meta import CropMeta
from preprocessing.pixel_normalizer import get_latest_total_mean
from preprocessing.sample import Sample
from utils.logger import Logger

logger = Logger('TESTING')

def mse(target_array, prediction_array, ind):
    standard_mse = torch.nn.MSELoss()
    if prediction_array.shape != target_array.shape:
        raise IndexError(
            f"Target shape is {target_array.shape} but prediction shape is {prediction_array.shape}. Prediction {ind}")
    prediction_array, target_array = np.asarray(prediction_array, np.float64), np.asarray(target_array, np.float64)
    other_test = np.mean((prediction_array - target_array) ** 2)
    #simple_test = standard_mse(torch.from_numpy(prediction_array), torch.from_numpy(target_array))
    return other_test


# create mean MSE for all test samples based on the prediction file and target file
def scoring(exp_name):
    exp_path = c.FOLDER_PATH_MAIN_EXPERIMENTS + os.sep + exp_name
    with open(exp_path + os.sep + c.PKL_PREDICTIONS, 'rb') as pfh:
        predictions = pkl.load(pfh)
    with open(c.FILE_SUBMISSION_TESTING_TARGETS, 'rb') as tfh:
        targets = pkl.load(tfh)

    mses = np.zeros(shape=len(predictions))
    ind = 0
    for target, prediction in zip(targets, predictions):
        mses[ind] = mse(target, prediction, ind)
        ind += 1
    mean_loss = np.mean(mses)
    logger.log(f'mean loss: {mean_loss}')
    with open(exp_path + os.sep + 'submission_results.txt', 'w+') as fh:
        print(f'loss: {mean_loss}', file=fh)
    return mean_loss


# test best model over all given test data from test data pickle file and save results for later scoring
def create_submission_predictions(experiment_name):
    with open(c.FILE_SUBMISSION_TESTING_RAW_DATA, 'rb') as f:
        test_set = load(f)

    with open(c.FILE_SUBMISSION_TESTING_TARGETS, 'rb') as tfh:
        targets = pkl.load(tfh)

    crop_sizes = test_set['crop_sizes']
    crop_centers = test_set['crop_centers']
    images = test_set['images']
    exp_path = c.FOLDER_PATH_MAIN_EXPERIMENTS + os.sep + experiment_name
    model = torch.load(exp_path + os.sep + c.PKL_BEST_MODEL)
    model = model.to(c.device)
    total_pixel_mean = get_latest_total_mean()
    outputs = list()

    for ind in range(len(images)):
        crop_size = crop_sizes[ind]
        crop_center = crop_centers[ind]
        image = images[ind]
        image = np.array(image, dtype='float64')
        image /= c.MAX_PIXEL_VALUE
        #TODO: image -= total_pixel_mean

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

        output = model(X, [crop_meta])
        cpu_output = output.cpu().detach()
        del output
        #cpu_output += total_pixel_mean
        cpu_output *= c.MAX_PIXEL_VALUE

        outputs.append(cpu_output[0, 0, :crop_meta.height, :crop_meta.width])
        print(f'>>> tested sample {ind + 1}')
    del model
    with open(exp_path + os.sep + c.PKL_PREDICTIONS, 'wb') as f:
        dump(outputs, f)