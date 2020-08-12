import config as c
from dataset.dataset_provider import DataSetProvider
from model.auto_encoder import AutoEncoder
from model.experiment import Experiment
from model.simple_conv_net import SimpleCnn
from model.submission import create_submission_predictions, scoring
from model.train import train
from preprocessing.preprocessing_pool import PreprocessingPool


# TODO: new scaling PLS!!!
def create_and_scale_data(process_nom=2):
    c.clear_folders()
    c.create_folders()

    prepro_pool = PreprocessingPool(process_nom=process_nom)
    prepro_pool.start()

    dsp = DataSetProvider(c.FOLDER_PATH_PREPROCESSING)
    dsp.scale_datasets(process_nom=process_nom)


def run_experiment(name, model, params):
    c.create_folders()
    exp = Experiment(name)
    exp.train_model(model, params)


def run_multiple_experiments():
    names = ['new_exp']
    params = [{'lr':1e-3, 'wd': 1e-5}]
    models = [AutoEncoder()]
    for ind in range(len(names)):
        run_experiment(names[ind], models[ind], params[ind])

#create_and_scale_data(16)
run_multiple_experiments()