import feature_flag as ff
import config as c
from dataset.dataset_provider import DataSetProvider
from model.final_submission import create_final_predictions
from model.types.basic_conv import BasicConv
from model.experiment import Experiment
from model.types.simple_conv_net import SimpleCnn
from preprocessing.preprocessing_pool import PreprocessingPool


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
    exp.train_model(model, params, max_epochs=3)


def run_multiple_experiments():
    names = ['simple_cnn_hl_2_b_3_md','simple_cnn_hl_3_b_3_md','simple_cnn_hl_4_b_3_md','simple_cnn_hl_5_b_3_md','simple_cnn_hl_6_b_3_md']
    params = [{'lr':1e-3, 'wd': 1e-5},{'lr':1e-3, 'wd': 1e-5},{'lr':1e-3, 'wd': 1e-5},{'lr':1e-3, 'wd': 1e-5},{'lr':1e-3, 'wd': 1e-5}]
    models = [SimpleCnn(n_hidden_layers=2),SimpleCnn(n_hidden_layers=3),SimpleCnn(n_hidden_layers=4),SimpleCnn(n_hidden_layers=5),SimpleCnn(n_hidden_layers=6)]
    for ind in range(len(names)):
        run_experiment(names[ind], models[ind], params[ind])

if ff.CREATE_DATA:
    create_and_scale_data(16)
if ff.TRAIN_MODELS:
    run_multiple_experiments()

c.create_folders()
create_final_predictions('simple_cnn_hl_6_b_3_md')