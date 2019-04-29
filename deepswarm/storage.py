# Copyright (c) 2019 Edvinas Byla
# Licensed under MIT License

import hashlib
import pickle

from datetime import datetime

from . import base_path, cfg, left_cost_is_better


class Storage:
    """Class responsible for backups and weight reuse."""

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
        """Loads existing backup or creates a new backup directory."""

        # If storage directory doesn't exist create one
        storage_path = base_path / 'saves'
        if not storage_path.exists():
            storage_path.mkdir()

        # Check if user specified save folder which should be used to load the data
        user_folder = cfg['save_folder']
        if user_folder is not None and (storage_path / user_folder).exists():
            self.current_path = storage_path / user_folder
            self.loaded_from_save = True
            # Store deepswarm object to backup
            self.backup = self.load_object(Storage.ITEM["BACKUP"])
            self.backup.storage.loaded_from_save = True
            return

        # Otherwise create a new directory
        directory_path = storage_path / datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        if not directory_path.exists():
            directory_path.mkdir()
            self.current_path = directory_path
            return

    def setup_directories(self):
        """Creates all the required directories."""

        for directory in Storage.DIR.values():
            directory_path = self.current_path / directory
            if not directory_path.exists():
                directory_path.mkdir()

    def perform_backup(self):
        """Saves DeepSwarm object to the backup directory."""

        self.save_object(self.deepswarm, Storage.ITEM["BACKUP"])

    def save_model(self, backend, model, path_hashes, cost):
        """Saves the model and adds its information to the dictionaries.

        Args:
            backend: Backend object.
            model: model which represents neural network structure.
            path_hashes [string]: list of hashes, where each hash represents a
            sub-path.
            cost: cost associated with the model.
        """

        sub_path_associated = False
        # The last element describes the whole path
        model_hash = path_hashes[-1]

        # For each sub-path find it's corresponding entry in hash table
        for path_hash in path_hashes:
            # Check if there already exists model for this sub-path
            existing_model_hash = self.path_lookup.get(path_hash)
            model_info = self.models.get(existing_model_hash)

            # If the old model is better then skip this sub-path
            if model_info is not None and left_cost_is_better(model_info[0], cost):
                continue

            # Otherwise associated this sub-path with a new model
            self.path_lookup[path_hash] = model_hash
            sub_path_associated = True

        # Save model on disk only if it was associated with some sub-path
        if sub_path_associated:
            # Add an entry to models dictionary
            self.models[model_hash] = (cost, 0)
            # Save to disk
            self.save_specified_model(backend, model_hash, model)

    def load_model(self, backend, path_hashes, path):
        """Loads model with the best weights.

        Args:
            backend: Backend object.
            path_hashes [string]: list of hashes, where each hash represents a
            sub-path.
            path [Node]: a path which represents the model.
        Returns:
            if the model exists returns a tuple containing model and its hash,
            otherwise returns a tuple containing None values.
        """

        # Go through all hashes backwards
        for idx, path_hash in enumerate(path_hashes[::-1]):
            # See if particular hash is associated with some model
            model_hash = self.path_lookup.get(path_hash)
            model_info = self.models.get(model_hash)

            # Don't reuse model if it hasn't improved for longer than allowed in patience
            if model_hash is not None and model_info[1] < cfg['reuse_patience']:
                model = self.load_specified_model(backend, model_hash)
                # If failed to load model, skip to next hash
                if model is None:
                    continue

                # If there is no difference between models, just return the old model,
                # otherwise create a new model by reusing the old model. Even though,
                # backend.reuse_model function could be called to handle both
                # cases, this approach saves some unnecessary computation
                new_model = model if idx == 0 else backend.reuse_model(model, path, idx)

                # We also return base model (a model which was used as a base to
                # create a new model) hash. This hash information is used later to
                # track if the base model is improving over time or is it stuck
                return (new_model, model_hash)
        return (None, None)

    def load_specified_model(self, backend, model_name):
        """Loads specified model using its name.

        Args:
            backend: Backend object.
            model_name: name of the model.
        Returns:
            model which represents neural network structure.
        """

        file_path = self.current_path / Storage.DIR["MODEL"] / model_name
        model = backend.load_model(file_path)
        return model

    def save_specified_model(self, backend, model_name, model):
        """Saves specified model using its name without and adding its information
        to the dictionaries.

        Args:
            backend: Backend object.
            model_name: name of the model.
            model: model which represents neural network structure.
        """

        save_path = self.current_path / Storage.DIR["MODEL"] / model_name
        backend.save_model(model, save_path)

    def record_model_performance(self, path_hash, cost):
        """Records how many times the model cost didn't improve.

        Args:
            path_hash: hash value associated with the model.
            cost: cost value associated with the model.
        """

        model_hash = self.path_lookup.get(path_hash)
        old_cost, no_improvements = self.models.get(model_hash)

        # If cost hasn't changed at all, increment no improvement count
        if old_cost is not None and old_cost == cost:
            self.models[model_hash] = (old_cost, (no_improvements + 1))

    def hash_path(self, path):
        """Takes a path and returns a tuple containing path description and
        list of sub-path hashes.

        Args:
            path [Node]: path which represents the model.
        Returns:
            tuple where the first element is a string representing the path
            description and the second element is a list of sub-path hashes.
        """

        hashes = []
        path_description = str(path[0])
        for node in path[1:]:
            path_description += ' -> %s' % (node)
            current_hash = hashlib.sha3_256(path_description.encode('utf-8')).hexdigest()
            hashes.append(current_hash)
        return (path_description, hashes)

    def save_object(self, data, name):
        """Saves given object to the object backup directory.

        Args:
            data: object that needs to be saved.
            name: string value representing the name of the object.
        """

        with open(self.current_path / Storage.DIR["OBJECT"] / name, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def load_object(self, name):
        """Load given object from the object backup directory.

        Args:
            name: string value representing the name of the object.
        Returns:
            object which has the same name as the given argument.
        """

        with open(self.current_path / Storage.DIR["OBJECT"] / name, 'rb') as f:
            data = pickle.load(f)
        return data
