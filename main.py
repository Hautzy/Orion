import config as c
from preprocessing.PreprocessingPool import PreprocessingPool

c.clear_folders()
c.create_folders()

prepro_pool = PreprocessingPool(process_nom=10)
prepro_pool.start()