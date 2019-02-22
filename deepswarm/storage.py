# Copyright (c) 2019 Edvinas Byla
# Licensed under MIT License

import hashlib
import os
import pickle
from datetime import datetime
from pathlib import Path
from . import cfg


class Storage:
    DIR = {
        "MODEL": "models",
        "OBJECT": "objects",
    }

    ITEM = {"BACKUP": "backup"}

    def __init__(self, deepswarm):
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
        user_folder = cfg['save_folder']
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

    def save_model(self, backend, model, path_hashes):
        # Last element describes whole path
        model_hash = path_hashes[-1]
        # Point each sub-path hash to main model hash
        for path_hash in path_hashes:
            self.model_cache[path_hash] = model_hash
        # Save main model on disk
        save_path = self.current_path / Storage.DIR["MODEL"] / model_hash
        backend.save_model(model, save_path)

    def load_model(self, backend, path_hashes, path):
        # Go trough all hashes backwards
        for idx, path_hash in enumerate(path_hashes[::-1]):
            # See if particular hash is associated with some model
            model_hash = self.model_cache.get(path_hash)
            if model_hash is not None:
                file_path = self.current_path / Storage.DIR["MODEL"] / model_hash
                model = backend.load_model(file_path)
                # If failed to load model, skip to next hash
                if model is None:
                    continue
                # If there is no difference between models, just return old model
                if idx == 0:
                    return model
                # Otherwise reused old model to create a new one
                else:
                    return backend.reuse_model(model, path, idx)
        return None

    def hash_path(self, path):
        hashes = []
        path_description = str(path[0])
        for node in path[1:]:
            path_description += ' -> %s' % (node)
            current_hash = hashlib.sha3_256(path_description.encode('utf-8')).hexdigest()
            hashes.append(current_hash)
        return (path_description, hashes)

    def save_object(self, data, name):
        with open(self.current_path / Storage.DIR["OBJECT"] / name, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def load_object(self, name):
        with open(self.current_path / Storage.DIR["OBJECT"] / name, 'rb') as f:
            data = pickle.load(f)
        return data
