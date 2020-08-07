from datetime import datetime

class Logger:
    def __init__(self, module_name, write_to_file=True):
        self.module_name = module_name

    def log(self, text):
        print(f'[{datetime.now()} | {self.module_name}]>>> {text}')