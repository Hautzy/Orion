import config
import numpy as np
from preprocessing.PreprocessingProcess import PreprocessingProcess


class PreprocessingPool():

    def __init__(self, process_nom=1):
        self.process_nom = process_nom
        self.workload_packages = self.create_workload_packages()
        self.processes = self.init_processes()

    def start(self):
        self.start_processes()
        self.join_processes()


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
        raw_data_image_paths = config.get_all_files_in_folder(config.FOLDER_PATH_RAW_DATA)
        data_length = len(raw_data_image_paths)
        package_size = int(np.ceil(data_length / self.process_nom))
        workload_packages = list()
        for i in range(0, data_length, package_size):
            workload_packages.append(raw_data_image_paths[i: i + package_size])
        return workload_packages
