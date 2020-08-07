class DataSetProvider:
    def __init__(self, dataset_metas, data_folder):
        self.dataset_metas = dataset_metas
        self.data_folder = data_folder

    def determine_datasets(self):
        pass



class DataSetMeta:
    def __init__(self, name, data_portion):
        self.name = name
        self.data_portion = data_portion