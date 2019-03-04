# Copyright (c) 2019 Edvinas Byla
# Licensed under MIT License

import random
import math
from . import cfg, left_cost_is_better
from .log import Log
from .nodes import (BatchNormalizationNode, Conv2DNode, DenseNode, DropoutNode, EndNode, FlattenNode, InputNode, NeighbourNode, Pool2DNode)


class ACO:
    def __init__(self, backend, storage):
        self.graph = Graph()
        self.current_depth = 0
        self.backend = backend
        self.storage = storage

    def search(self):
        """ Performs ant colony system optimization over the graph.

        Returns:
            ant which found best network topology
        """

        # Generate random ant only if search started from zero
        if not self.storage.loaded_from_save:
            Log.header("STARTING ACO SEARCH", type="GREEN")
            self.best_ant = Ant(self.graph.generate_path(self.random_select))
            self.best_ant.evaluate(self.backend, self.storage)
            Log.info(self.best_ant)
        else:
            Log.header("RESUMING ACO SEARCH", type="GREEN")

        while self.graph.current_depth <= cfg['max_depth']:
            Log.header("Current search depth is %i" % self.graph.current_depth, type="GREEN")
            ants = self.generate_ants()
            # Sort ants depending on user selected metric
            ants.sort() if cfg['metrics'] == 'loss' else ants.sort(reverse=True)
            # If any of the new solutions has lower cost than best solution, update best
            if left_cost_is_better(ants[0].cost, self.best_ant.cost):
                self.best_ant = ants[0]
                Log.header("NEW BEST ANT FOUND", type="GREEN")

            Log.header("BEST ANT DURING ITERATION")
            Log.info(self.best_ant)
            # Do global pheromone update
            self.update_pheromone(ant=self.best_ant, update_rule=self.global_update)
            # Print pheromone information and increase graph's depth
            self.graph.show_pheromone()
            self.graph.increase_depth()
            # Do a backup
            self.storage.perform_backup()
        return self.best_ant

    def generate_ants(self):
        ants = []
        for ant_number in range(cfg['aco']['ant_count']):
            Log.header("GENERATING ANT %i" % (ant_number + 1))
            ant = Ant()
            # Generate ant's path using given ACO rule
            ant.path = self.graph.generate_path(self.aco_select)
            # Evaluate how good is new path
            ant.evaluate(self.backend, self.storage)
            ants.append(ant)
            Log.info(ant)
            self.update_pheromone(ant=ant, update_rule=self.local_update)
        return ants

    def random_select(self, neighbours):
        current_node = random.choice(neighbours).node
        current_node.select_random_attributes()
        return current_node

    def aco_select(self, neighbours):
        # Transform list of NeighbourNode objects to list of tuples (Node, pheromone, heuristic)
        tuple_neighbours = [(n.node, n.pheromone, n.heuristic) for n in neighbours]
        # Select node using ant colony select rule
        current_node = self.aco_select_rule(tuple_neighbours)
        # Select custom attributes using ant colony select rule
        current_node.select_custom_attributes(self.aco_select_rule)
        return current_node

    def aco_select_rule(self, neighbours):
        """Selects neighbour node based on ant colony system transition rule

        Args:
            neighbours [(Object, float, float)]: list of tuples, where each tuple
            containts object to be selected object's pheromone value and object's heuristic value
        Returns:
            selected object
        """
        probabilities = []
        denominator = 0.0

        for (_, pheromone, heuristic) in neighbours:
            probability = pheromone * heuristic
            probabilities.append(probability)
            denominator += probability
        # Try to perform greedy select: exploitation
        random_variable = random.uniform(0, 1)
        if random_variable <= cfg['aco']['greediness']:
            # Find max probability
            max_probability = max(probabilities)
            # Gather indices of probabilities that are equal to max probability
            max_indices = [i for i, j in enumerate(probabilities) if j == max_probability]
            # From those max indices select random index
            neighbour_index = random.choice(max_indices)
            return neighbours[neighbour_index][0]
        # Otherwise perform select using roulette wheel: exploration
        probabilities = [x / denominator for x in probabilities]
        probability_sum = sum(probabilities)
        random_treshold = random.uniform(0, probability_sum)
        current_value = 0
        for neighbour_index, probability in enumerate(probabilities):
            current_value += probability
            if current_value > random_treshold:
                return neighbours[neighbour_index][0]

    def update_pheromone(self, ant, update_rule):
        current_node = self.graph.input_node
        # Skip input node as it's not connected to any previous node
        for node in ant.path[1:]:
            # Use node from path to retrieve its corresponding node in graph
            neighbour = next((x for x in current_node.neighbours if type(x.node) is type(node)), None)
            # If path was closed using complete_path method, ignore rest of the path
            if neighbour is None:
                break
            # Update pheromone connecting to neighbour
            neighbour.pheromone = update_rule(
                old_value=neighbour.pheromone,
                cost=ant.cost
            )
            # Update attribute pheromone values
            for attribute in neighbour.node.attributes:
                # Find what attribute value was used for node
                attribute_value = getattr(node, attribute.name)
                # Retrieve pheromone for that value
                old_pheromone_value = attribute.dict[attribute_value]
                # Update pheromone
                attribute.dict[attribute_value] = update_rule(
                    old_value=old_pheromone_value,
                    cost=ant.cost
                )
            # Advance current node
            current_node = neighbour.node

    def local_update(self, old_value, cost):
        decay = cfg['aco']['pheromone']['decay']
        pheromone_0 = cfg['aco']['pheromone']['start']
        return (1 - decay) * old_value + (decay * pheromone_0)

    def global_update(self, old_value, cost):
        # Calculate solution cost based on metrics
        added_pheromone = (1 / (cost * 10)) if cfg['metrics'] == 'loss' else cost
        evaporation = cfg['aco']['pheromone']['evaporation']
        return (1 - evaporation) * old_value + (evaporation * added_pheromone)


class Ant:
    def __init__(self, path=[]):
        self.path = path
        self.loss = math.inf
        self.accuracy = 0.0
        self.path_description = None
        self.path_hash = None

    @property
    def cost(self):
        return self.loss if cfg['metrics'] == 'loss' else self.accuracy

    def __lt__(self, other):
        return self.cost < other.cost

    def __str__(self):
        return "======= \n Ant: %s \n Loss: %f \n Accuracy: %f \n Path: %s \n Hash: %s \n=======" % (
            hex(id(self)),
            self.loss,
            self.accuracy,
            self.path_description,
            self.path_hash,
        )

    def evaluate(self, backend, storage):
        # Extract path information
        self.path_description, path_hashes = storage.hash_path(self.path)
        self.path_hash = path_hashes[-1]
        # Check if model already exists if yes, then just re-use it
        existing_model, existing_model_hash = storage.load_model(backend, path_hashes, self.path)
        if existing_model is None:
            # Generate model
            new_model = backend.generate_model(self.path)
        else:
            # Re-use model
            new_model = existing_model
        # Train model
        new_model = backend.train_model(new_model)
        # Evaluate model
        self.loss, self.accuracy = backend.evaluate_model(new_model)
        # If new model was created from older model, record older model progress
        if existing_model_hash is not None:
            storage.record_model_performance(existing_model_hash, self.cost)
        # Save model
        storage.save_model(backend, new_model, path_hashes, self.cost)


class Graph:
    def __init__(self, current_depth=0):
        self.topology = []
        self.current_depth = current_depth
        self.input_node = self.get_node(InputNode(), current_depth)
        self.increase_depth()

    def get_node(self, node, depth):
        # If we are trying to insert node into not existing layer, we pad topology
        # by adding empty dictionaries, untill required depth is reached
        while depth > (len(self.topology) - 1):
            self.topology.append({})

        # If node already exists return it, otherwise add it to topology first
        return self.topology[depth].setdefault(node.name, node)

    def increase_depth(self):
        self.current_depth += 1

    def generate_path(self, select_rule):
        """Generates path trough the graph, based on given rule

        Args:
            select_rule ([Node]): function which receives list of neighbours

        Returns:
            path which contains Node objects
        """
        current_node = self.input_node
        path = [current_node.create_deepcopy()]
        for depth in range(self.current_depth):
            # If node doesn't have any neigbours stop expanding path
            if not self.has_neighbours(current_node, depth):
                break
            # Select node using rule
            current_node = select_rule(current_node.neighbours)
            # Add only copy of the node, so that original stays unmodified
            path.append(current_node.create_deepcopy())
        completed_path = self.complete_path(path)
        return completed_path

    def has_neighbours(self, node, depth):
        # Expand only if it haven't been expanded
        if node.is_expanded is False:
            available_transitions = node.available_transitions
            for (transition_class, heuristic_value) in available_transitions:
                neighbour_node = self.get_node(transition_class(), depth + 1)
                node.neighbours.append(NeighbourNode(neighbour_node, heuristic_value))
            node.is_expanded = True
        # Return value indicating if node has neigbours after beign expanded
        return len(node.neighbours) > 0

    def complete_path(self, path):
        # If path is already completed then return that path
        if type(path[-1]) is EndNode:
            return path
        # Otherwise complete the path and then return completed path
        # We intentionally don't add these eding nodes as neighbours to the last node
        # in the path, because during first few iteration these nodes will always be part
        # of the best path (as it's impossible to close path automatically when it's so short)
        # this would result in bias pheromone received by these nodes during later iterations
        if type(path[-1]) in (BatchNormalizationNode, Conv2DNode, Pool2DNode):
            path.append(self.get_node(FlattenNode(), len(path)))
        if type(path[-1]) in (FlattenNode, DenseNode, DropoutNode):
            path.append(self.get_node(EndNode(), len(path)))
        return path

    def show_pheromone(self):
        Log.header("PHEROMONE START", type="RED")
        for idx, layer in enumerate(self.topology):
            info = []
            for node in layer.values():
                for neighbour in node.neighbours:
                    info.append("%s [%s] -> %f -> %s [%s]" % (node.name, hex(id(node)),
                        neighbour.pheromone, neighbour.node.name, hex(id(neighbour.node))))
                    # If neighbour node doesn't have any attributes skip attribute info
                    if not neighbour.node.attributes:
                        continue
                    info.append("\t%s [%s]:" % (neighbour.node.name, hex(id(neighbour.node))))
                    for attribute in neighbour.node.attributes:
                        info.append("\t\t%s: %s" % (attribute.name, attribute.dict))
            if info:
                Log.header("Layer %d" % (idx + 1))
                Log.info('\n'.join(info))
        Log.header("PHEROMONE END", type="RED")
