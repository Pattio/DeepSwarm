# Copyright (c) 2019 Edvinas Byla
# Licensed under MIT License

from .aco import ACO
from .log import Log
from .storage import Storage

from . import settings


class DeepSwarm:
    def __init__(self, backend):
        self.backend = backend
        self.storage = Storage(self)
        # Enable logging and log current settings
        self.setup_logging()
        # Try to load from backup
        if self.storage.loaded_from_save:
            self.__dict__ = self.storage.backup.__dict__

    def setup_logging(self):
        Log.enable(self.storage)
        Log.header("DeepSwarm settings")
        Log.info(settings)

    def find_topology(self):
        """Finds neural network topology which has lowest loss

        Args:
            max_depth (int): maximum number of hidden layers
            swarm_size(int): number of ants, which are searching for the topology
        Returns:
            network model in the format of backend which was used during initialization
        """
        # Create new object only if there are no backups
        if not self.storage.loaded_from_save:
            self.aco = ACO(backend=self.backend, storage=self.storage)

        best_ant = self.aco.search()
        best_model = self.storage.load_specified_model(self.backend, best_ant.path_hash)
        return best_model
