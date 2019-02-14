# Copyright (c) 2019 Edvinas Byla
# Licensed under MIT License

import os
import pickle
from datetime import datetime
from pathlib import Path
from . import config as cfg


class Storage:
    DIR = {
        "MODEL": "models",
        "OBJECT": "objects",
    }

    ITEM = {
        "BACKUP": "backup"
    }

    def __init__(self, deepswarm):
        self.current_path = None
        self.loaded_from_save = False
        self.backup = None
        self.model_cache = {}
        self.deepswarm = deepswarm
        self.setup_path()
        self.setup_directories()

    def setup_path(self):
        base_path = Path(os.path.dirname(os.path.dirname(__file__))) / "saves"
        # If storage folder doesn't exist create one
        if not base_path.exists():
            base_path.mkdir()
        # Check if user specified save folder, which should be used to load data
        user_folder = cfg.SAVE_FOLDER
        if user_folder is not None and (base_path / user_folder).exists():
            self.current_path = base_path / user_folder
            self.loaded_from_save = True
            # Store deepswarm object to backup
            self.backup = self.load_object(Storage.ITEM["BACKUP"])
            self.backup.storage.loaded_from_save = True
            return
        # Otherwise create new directory
        directory_path = base_path / datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        if not directory_path.exists():
            directory_path.mkdir()
            self.current_path = directory_path
            return

    def setup_directories(self):
        for directory in Storage.DIR.values():
            directory_path = self.current_path / directory
            if not directory_path.exists():
                directory_path.mkdir()

    def perform_backup(self):
        self.save_object(self.deepswarm, Storage.ITEM["BACKUP"])

    def save_model(self, backend, model, model_name):
        path = self.current_path / Storage.DIR["MODEL"] / model_name
        backend.save_model(model, path)
        # Add record to dictioanry, which indicates that model should be on disk
        self.model_cache[model_name] = True

    def load_model(self, backend, model_name):
        # Check disk only if model was in dictionary cache
        if model_name in self.model_cache:
            path = self.current_path / Storage.DIR["MODEL"] / model_name
            return backend.load_model(path)
        else:
            return None

    def save_object(self, data, name):
        with open(self.current_path / Storage.DIR["OBJECT"] / name, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def load_object(self, name):
        with open(self.current_path / Storage.DIR["OBJECT"] / name, 'rb') as f:
            data = pickle.load(f)
        return data
