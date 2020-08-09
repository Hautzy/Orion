import os
from multiprocessing import Process, Manager

import numpy as np

import config as c
from utils.divide_and_conquer_pool import DivideAndConquerPool
from utils.logger import Logger


def read_all_uuids():
    with open(c.FILE_TOTAL_IMAGE_CROP_META_CROSS_REFERENCE, 'r') as f:
        uuids = [line.split(';')[0] for line in f.readlines()]
    return uuids


class PixelNormalizer(DivideAndConquerPool):

    def __init__(self, indices, process_nom):
        super(PixelNormalizer, self).__init__(PixelNormalizer.remove_duplicate_uuid_indices(indices), process_nom)
        self.logger = Logger(f'PIXELNORMALIZER')
        self.processes = self.init_fit_processes()
        self.total_mean = 0.0

    def init_fit_processes(self):
        processes = list()
        for i in range(self.process_nom):
            new_process = FitProcess(i, self.workload_packages[i])
            processes.append(new_process)
        return processes

    def init_transform_processes(self):
        self.processes = list()
        for i in range(self.process_nom):
            new_process = TransformProcess(i, self.total_mean, self.workload_packages[i])
            self.processes.append(new_process)

    def combine_results(self):
        total_means = np.zeros(shape=self.process_nom)
        for i, process in enumerate(self.processes):
            with open(f'{c.FOLDER_PATH_PREPROCESSING}{os.sep}{process.id}{c.FILE_PART_PART_IMAGE_MEAN}', 'rb') as f:
                total_means[i] = np.load(f)[0]
        self.total_mean = np.mean(total_means)
        mean_result = np.full(shape=1, fill_value=self.total_mean)
        with open(c.FILE_TOTAL_IMAGE_MEAN, 'wb') as f:
            np.save(f, mean_result)
        self.logger.log(f'total mean {self.total_mean}')

    def start_fit(self):
        self.logger.log('start fitting normalizer')
        self.start()
        self.logger.log('finished fitting normalizer')

    def start_transform(self, sample_nom):
        paths = c.get_all_files_in_folder(c.FOLDER_PATH_PREPROCESSING, lambda f: '.npy' in f)
        self.workload_packages = self.create_workload_packages(paths)
        self.init_transform_processes()
        self.logger.log('start transforming all images')
        self.start_processes()
        self.join_processes()
        self.logger.log('finished transforming all images')

    @staticmethod
    def remove_duplicate_uuid_indices(indices):
        uuids = read_all_uuids()
        checked_uuids = list()
        unique_uuids_indices = list()
        for ind in indices:
            current_uuid = uuids[ind]
            if current_uuid not in checked_uuids:
                checked_uuids.append(current_uuid)
                unique_uuids_indices.append(ind)
        return unique_uuids_indices


class FitProcess(Process):
    def __init__(self, idx, workload):
        super(FitProcess, self).__init__()
        self.id = idx
        self.workload = workload
        self.workload_length = len(self.workload)
        self.logger = Logger(f'FITPROCESS-{self.id}')

    def run(self):
        uuids = read_all_uuids()
        uuids = set(uuids[ind] for ind in self.workload)
        image_length = len(uuids)
        means = np.zeros(shape=image_length)
        for i, image_uuid in enumerate(uuids):
            with open(f'{c.FOLDER_PATH_PREPROCESSING}{os.sep}{image_uuid}.npy', 'rb') as f:
                image_data = np.load(f)
                means[i] = np.mean(image_data)
        total_mean = np.mean(means)
        mean_result = np.full(shape=1, fill_value=total_mean)
        with open(f'{c.FOLDER_PATH_PREPROCESSING}{os.sep}{self.id}{c.FILE_PART_PART_IMAGE_MEAN}', 'wb') as f:
            np.save(f, mean_result)
        self.logger.log(f'total mean: {total_mean} from {image_length} images')


class TransformProcess(Process):
    def __init__(self, idx, total_mean, workload):
        super(TransformProcess, self).__init__()
        self.id = idx
        self.workload = workload
        self.total_mean = total_mean
        self.workload_length = len(self.workload)
        self.logger = Logger(f'TRANSFORMPROCESS-{self.id}')

    def run(self):
        cnt = 0
        for i, path in enumerate(self.workload):
            with open(path, 'rb') as f:
                image_data = np.load(f)
                image_data -= self.total_mean
            with open(path, 'wb') as f:
                np.save(f, image_data)
            cnt += 1
        self.logger.log(f'transformed {cnt} images')
