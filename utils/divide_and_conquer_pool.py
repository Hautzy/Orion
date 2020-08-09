import os

import numpy as np

import config as c


class DivideAndConquerPool():

    def __init__(self, workload_data, process_nom=1):
        self.process_nom = process_nom
        self.workload_packages = self.create_workload_packages(workload_data)
        self.processes = list()

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

    def create_workload_packages(self, workload_data):
        data_length = len(workload_data)
        package_size = int(np.ceil(data_length / self.process_nom))
        workload_packages = list()
        for i in range(0, data_length, package_size):
            workload_packages.append(workload_data[i: i + package_size])
        return workload_packages

    def combine_results(self):
        pass
