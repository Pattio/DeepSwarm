# Copyright (c) 2019 Edvinas Byla
# Licensed under MIT License

from .aco import ACO


class DeepSwarm:
    def __init__(self, backend):
        self.backend = backend

    def find_topology(self, max_depth, swarm_size):
        """Finds neural network topology which has lowest loss

        Args:
            max_depth (int): maximum number of hidden layers
            swarm_size(int): number of ants, which are searching for the topology
        Returns:
            network model in the format of backend which was used during initialization
        """
        aco = ACO(
            max_iteration=max_depth,
            ants_number=swarm_size,
            backend=self.backend
        )
        best_ant = aco.search()
        return best_ant.model
