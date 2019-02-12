# Copyright (c) 2019 Edvinas Byla
# Licensed under MIT License

import os
import pickle
from datetime import datetime
from . import config as cfg


class Storage:
    def __init__(self, location=None):
        self.base_path = os.path.dirname(os.path.dirname(__file__)) + "/saves/"
        self.current_path = None
        self.load_from_save = False
        self.setup_path(location)

    def setup_path(self, location):
        # If storage folder doesn't exist create one
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
        # Check if user specified save folder, which should be used to load data
        if cfg.SAVE_FOLDER is not None and os.path.isdir(self.base_path + cfg.SAVE_FOLDER):
            self.current_path = self.base_path + cfg.SAVE_FOLDER
            self.load_from_save = True
            return
        # Otherwise create new directory
        directory_path = self.base_path + datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            self.current_path = directory_path
            return

    def save(self, data, name):
        with open(self.current_path + '/' + name, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def load(self, name):
        with open(self.current_path + '/' + name, 'rb') as f:
            data = pickle.load(f)
        return data
