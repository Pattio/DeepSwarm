# Copyright (c) 2019 Edvinas Byla
# Licensed under MIT License

import hashlib
import pickle
from datetime import datetime
from . import base_path, cfg, left_cost_is_better


class Storage:
    DIR = {
        "MODEL": "models",
        "OBJECT": "objects",
    }

    ITEM = {"BACKUP": "backup"}

    def __init__(self, deepswarm):
        self.loaded_from_save = False
        self.backup = None
        self.path_lookup = {}
        self.models = {}
        self.deepswarm = deepswarm
        self.setup_path()
        self.setup_directories()

    def setup_path(self):
        storage_path = base_path / 'saves'
        # If storage folder doesn't exist create one
        if not storage_path.exists():
            storage_path.mkdir()
        # Check if user specified save folder, which should be used to load data
        user_folder = cfg['save_folder']
        if user_folder is not None and (storage_path / user_folder).exists():
            self.current_path = storage_path / user_folder
            self.loaded_from_save = True
            # Store deepswarm object to backup
            self.backup = self.load_object(Storage.ITEM["BACKUP"])
            self.backup.storage.loaded_from_save = True
            return
        # Otherwise create new directory
        directory_path = storage_path / datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
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

    def save_model(self, backend, model, path_hashes, cost):
        sub_path_associated = False
        # Last element describes whole path
        model_hash = path_hashes[-1]
        # For each sub-path find it's correpsonding entry in hash table
        for path_hash in path_hashes:
            # Check if there already exists model for this sub-path
            existing_model_hash = self.path_lookup.get(path_hash)
            model_info = self.models.get(existing_model_hash)
            # If old model is better then skip this sub-path
            if model_info is not None and left_cost_is_better(model_info[0], cost):
                continue
            # Otherwise associated this sub-path with new model
            self.path_lookup[path_hash] = model_hash
            sub_path_associated = True

        # Save model on disk only if it was associated with some sub-path
        if sub_path_associated:
            # Add entry to models dictionary
            self.models[model_hash] = (cost, 0)
            # Save to disk
            self.save_specified_model(backend, model_hash, model)

    def load_model(self, backend, path_hashes, path):
        # Go trough all hashes backwards
        for idx, path_hash in enumerate(path_hashes[::-1]):
            # See if particular hash is associated with some model
            model_hash = self.path_lookup.get(path_hash)
            model_info = self.models.get(model_hash)
            # Don't reuse model if it haven't improved for longer than allowed in patience
            if model_hash is not None and model_info[1] < cfg['reuse_patience']:
                model = self.load_specified_model(backend, model_hash)
                # If failed to load model, skip to next hash
                if model is None:
                    continue
                # If there is no difference between models, just return old model,
                # otherwise create a new model by reusing old model. Even though,
                # backend.reuse_model function could be called to handle both
                # cases, this approach saves some unnecessary computation
                new_model = model if idx == 0 else backend.reuse_model(model, path, idx)
                # We also return base model (model which was used as a base to
                # create new model) hash. This hash information is used later to
                # track if base model is improving over time or is it stuck
                return (new_model, model_hash)
        return (None, None)

    def load_specified_model(self, backend, model_name):
        file_path = self.current_path / Storage.DIR["MODEL"] / model_name
        model = backend.load_model(file_path)
        return model

    def save_specified_model(self, backend, model_name, model):
        save_path = self.current_path / Storage.DIR["MODEL"] / model_name
        backend.save_model(model, save_path)

    def record_model_performance(self, path_hash, cost):
        model_hash = self.path_lookup.get(path_hash)
        old_cost, no_improvements = self.models.get(model_hash)
        # If cost haven't changed at all, increment no improvement count
        if old_cost is not None and old_cost == cost:
            self.models[model_hash] = (old_cost, (no_improvements + 1))

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
