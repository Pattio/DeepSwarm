# Copyright (c) 2019 Edvinas Byla
# Licensed under MIT License

import random
import math
from .graph import Graph


class ACO:
    def __init__(self, max_iteration, ants_number, backend):
        self.graph = Graph()
        self.max_iteration = max_iteration
        self.ants_number = ants_number
        self.backend = backend

        self.greediness = 0.5
        self.pheromone = Pheromone(
            pheromone_0=0.1,
            decay=0.1,
            evaporation=0.1,
        )

    def search(self):
        """ Performs ant colony system optimization over the graph.

        Returns:
            ant which found best network topology
        """
        best_ant = Ant(self.graph.generate_random_path())
        best_ant.evaluate(self.backend)

        for _ in range(self.max_iteration):
            ants = self.generate_ants()
            ants.sort()
            # If any of the new solutions has lower cost than best solution, update best
            if ants[0].cost < best_ant.cost:
                best_ant = ants[0]

            # Do global pheromone update
            self.pheromone.update(ant=best_ant, local=False)
            # Expand graph
            self.graph.increase_depth()
        return best_ant

    def generate_ants(self):
        ants = []
        for _ in range(self.ants_number):
            ant = Ant()
            # Generate ant's path based on pheremone
            ant.path = self.generate_path()
            # TODO: Check if path is unique if not then don't evaluate this ant
            # and use stats from already evaluated ant
            ant.evaluate(self.backend)
            ants.append(ant)
            self.pheromone.update(ant=ant, local=True)
        return ants

    def generate_path(self):
        """Generates path trough the graph, based on ACO rules

        Returns:
            path which contains Node objects
        """
        path = [self.graph.input_node.create_deepcopy()]
        path_identifier = path[0].name
        current_node = self.graph.input_node

        for _ in range(self.graph.current_depth):
            current_node.expand()
            # Stop expanding node, when it can't be expanded any more
            if not current_node.neighbours:
                break
            # Select neighbour and update current_node to be that neighbour
            neighbour_index = self.select_neighbour(current_node, path_identifier)
            current_node = current_node.neighbours[neighbour_index]
            # Update path
            path.append(current_node.create_deepcopy())
            path_identifier += current_node.name
        completed_path = self.graph.complete_path(path)
        return completed_path

    def select_neighbour(self, current_node, path_identifier):
        """Selects neighbour node based on ant colony system transition probability

        Args:
            current_node (Node): node from which next node should be selected
            path_identifier(String): string which describes path to the current_node
        Returns:
            index of the neighbour which was selected
        """
        probabilities = []
        denominator = 0.0
        for neighbour in current_node.neighbours:
            neighbour_pheromone = self.pheromone.get(path_identifier + neighbour.name)
            probabilities.append(neighbour_pheromone)
            denominator += neighbour_pheromone
        # Try to perform greedy select - exploitation
        random_variable = random.uniform(0, 1)
        if random_variable <= self.greediness:
            # do greedy select
            max_probability = max(probabilities)
            max_indices = [i for i, j in enumerate(probabilities) if j == max_probability]
            neighbour_index = random.choice(max_indices)
            return neighbour_index
        # Otherwise perform select using roulette wheel - exploration
        probabilities = [x / denominator for x in probabilities]
        probability_sum = sum(probabilities)
        random_treshold = random.uniform(0, probability_sum)
        current_value = 0
        for idx, probability in enumerate(probabilities):
            current_value += probability
            if current_value > random_treshold:
                return idx


class Ant:
    def __init__(self, path=[]):
        self.path = path
        self.cost = math.inf
        self.model = None

    def __lt__(self, other):
        return self.cost < other.cost

    def evaluate(self, backend):
        self.model = backend.generate_model(self.path)
        self.cost = backend.evaluate_model(self.model)


class Pheromone():
    # TODO: introduce pheromone importance
    def __init__(self, pheromone_0, decay, evaporation):
        self.pheromone_0 = pheromone_0
        self.decay = decay
        self.evaporation = evaporation
        self.pheromone_matrix = {}

    def get(self, identifier):
        """Retrieves pheromone amount for specified identifier.

        If edge doesn't exists yet, new edge with start pheromone value is created

        Args:
            identifier (string): identifier, which describes edge, this edge is
            describied by using full path i.e. InputNodeConv2DNode3FlattenNode
        Returns:
            amount of pheromone on the last edge i.e if path was
            InputNodeConv2DNode3FlattenNode then returned number describes
            pheromone amount on edge Conv2DNode3-FlattenNode

        """
        # TODO: Generate hash value to save memory space
        return self.pheromone_matrix.setdefault(identifier, self.pheromone_0)

    def generate_identifiers(self, path):
        """Generates identifier for each path element

        Args:
            path ([Node]): list of nodes describing path
        Returns:
            list of identifiers, where i-th element corresponds to i-th node identifier
        """
        path_identifier = ""
        path_identifiers = []
        for node in path:
            # Append each node's name to identifier to generate unique string
            path_identifier += node.name
            path_identifiers.append(path_identifier)
        return path_identifiers

    def exists(self, path):
        '''
        TODO: refactor method, because sometimes even though identifier exists
        it doesn't mean that it was evaluated. For example if you select neighbour,
        identifiers for all nodes will be created, however not all paths will be
        evaluated.
        '''

        # Generate identifiers from path
        identifiers = self.generate_identifiers(path)
        # Check if last element, which describes whole path is already in matrix
        return identifiers[-1] in self.pheromone_matrix

    def update(self, ant, local):
        # Generate edge identifiers for path
        identifiers = self.generate_identifiers(ant.path)
        # Update pheromone value for each edge
        for identifier in identifiers:
            old_value = self.get(identifier)
            if local:
                new_value = (1 - self.decay) * old_value + (self.decay * self.pheromone_0)
            else:
                new_value = (1 - self.evaporation) * old_value + (self.evaporation * (1 / (ant.cost * 10)))
            self.pheromone_matrix[identifier] = new_value
