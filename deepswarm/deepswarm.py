# Copyright (c) 2019 Edvinas Byla
# Licensed under MIT License

from . import settings, left_cost_is_better
from .aco import ACO
from .log import Log
from .storage import Storage


class DeepSwarm:
    """Class responsible for providing user facing interface."""

    def __init__(self, backend):
        self.backend = backend
        self.storage = Storage(self)

        # Enable logging and log current settings
        self.setup_logging()

        # Try to load from the backup
        if self.storage.loaded_from_save:
            self.__dict__ = self.storage.backup.__dict__

    def setup_logging(self):
        """Enables logging and logs current settings."""

        Log.enable(self.storage)
        Log.header("DeepSwarm settings")
        Log.info(settings)

    def find_topology(self):
        """Finds the best neural network topology.

        Returns:
            network model in the format of backend which was used during
            initialization.
        """

        # Create a new object only if there are no backups
        if not self.storage.loaded_from_save:
            self.aco = ACO(backend=self.backend, storage=self.storage)

        best_ant = self.aco.search()
        best_model = self.storage.load_specified_model(self.backend, best_ant.path_hash)
        return best_model

    def train_topology(self, model, epochs, augment):
        """Trains given neural network topology for a specified number of epochs.

        Args:
            model: model which represents neural network structure.
            epochs (int): for how many epoch train the model.
            augment (kwargs): augmentation arguments.
        Returns:
            network model in the format of backend which was used during
            initialization.
        """

        # Before training make a copy of old weights in case performance
        # degrades during the training
        loss, accuracy = self.backend.evaluate_model(model)
        old_weights = model.get_weights()

        # Train the network
        model_name = 'best-trained-topology'
        trained_topology = self.backend.fully_train_model(model, epochs, augment)
        loss_new, accuracy_new = self.backend.evaluate_model(trained_topology)

        # Setup the metrics
        if settings['DeepSwarm']['metrics'] == 'loss':
            metrics_old = loss
            metrics_new = loss_new
        else:
            metrics_old = accuracy
            metrics_new = accuracy_new

        # Restore the weights if performance did not improve
        if left_cost_is_better(metrics_old, metrics_new):
            trained_topology.set_weights(old_weights)

        # Save and return the best topology
        self.storage.save_specified_model(self.backend, model_name, trained_topology)
        return self.storage.load_specified_model(self.backend, model_name)

    def evaluate_topology(self, model):
        """Evaluates neural network performance."""

        Log.header('EVALUATING PERFORMANCE ON TEST SET')
        loss, accuracy = self.backend.evaluate_model(model)
        Log.info('Accuracy is %f and loss is %f' % (accuracy, loss))
