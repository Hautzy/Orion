import config as c
from dataset.dataset_provider import DataSetProvider
from model.evaluation import plot_experiment
from model.simple_conv_net import SimpleCnn
from model.submission import create_submission_predictions, scoring
from model.train import train
from preprocessing.preprocessing_pool import PreprocessingPool


def create_and_scale_data(process_nom=2):
    c.clear_folders()
    c.create_folders()

    prepro_pool = PreprocessingPool(process_nom=process_nom)
    prepro_pool.start()

    dsp = DataSetProvider(c.FOLDER_PATH_PREPROCESSING)
    dsp.scale_datasets(process_nom=process_nom)


def start_experiment(params):
    exp_name = params['exp_name']
    c.create_folders()
    train(exp_name, params)
    create_submission_predictions(exp_name)
    scoring(exp_name)

def only_submission(exp_name):
    create_submission_predictions(exp_name)
    scoring(exp_name)


#create_and_scale_data(16)

params = [{
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'model': SimpleCnn(1),
    'exp_name': 'exp_01'
    }, {
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'model': SimpleCnn(2),
        'exp_name': 'exp_02'
    }, {
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'model': SimpleCnn(3),
        'exp_name': 'exp_03'
    }, {
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'model': SimpleCnn(4),
        'exp_name': 'exp_04'
    }, {
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'model': SimpleCnn(5),
        'exp_name': 'exp_05'
    }, {
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'model': SimpleCnn(6),
        'exp_name': 'exp_06'
    }, {
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'model': SimpleCnn(7),
        'exp_name': 'exp_07'
    }]

#for param in params:
    #start_experiment(param)
