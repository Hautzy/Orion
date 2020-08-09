import os

import numpy as np

import config as c
from preprocessing.preprocessing_process import PreprocessingProcess


class PreprocessingPool():

    def __init__(self, process_nom=1):
        self.process_nom = process_nom
        self.workload_packages = self.create_workload_packages()
        self.processes = self.init_processes()

    def start(self):
        self.start_processes()
        self.join_processes()
        self.combine_results()


    def start_processes(self):
        for process in self.processes:
            process.start()

    def join_processes(self):
        for process in self.processes:
            process.join()

    def init_processes(self):
        processes = list()
        for i in range(self.process_nom):
            new_process = PreprocessingProcess(i, self.workload_packages[i])
            processes.append(new_process)
        return processes

    def create_workload_packages(self):
        raw_data_image_paths = c.get_all_files_in_folder(c.FOLDER_PATH_RAW_DATA)
        data_length = len(raw_data_image_paths)
        package_size = int(np.ceil(data_length / self.process_nom))
        workload_packages = list()
        for i in range(0, data_length, package_size):
            workload_packages.append(raw_data_image_paths[i: i + package_size])
        return workload_packages

    def combine_results(self):
        with open(c.FILE_TOTAL_IMAGE_CROP_META_CROSS_REFERENCE, 'w') as total_file:
            for i, process in enumerate(self.processes):
                idx = process.id
                with open(f'{c.FOLDER_PATH_PREPROCESSING}{os.sep}{idx}{c.FILE_PART_IMAGE_CROP_META_CORSS_REFERENCE}', 'r') as part_file:
                    total_file.writelines(part_file.readlines())

