# Copyright (c) 2019 Edvinas Byla
# Licensed under MIT License

import random
from .nodes import (Conv2DNode, DenseNode, DropoutNode, EndNode, FlattenNode, InputNode, Pool2DNode)


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

    def generate_random_path(self):
        current_node = self.input_node
        path = [current_node.create_deepcopy()]
        for depth in range(self.current_depth):
            # Expand only if it haven't been expanded
            if current_node.is_expanded is False:
                available_transitions = current_node.available_transitions
                for available_transition in available_transitions:
                    neighbour_node = self.get_node(available_transition(), depth + 1)
                    current_node.neighbours.append(neighbour_node)
                current_node.is_expanded = True
            # Select new random node and append it to the path
            current_node = random.choice(current_node.neighbours)
            current_node.select_random_attributes()
            # Add only copy of the node, so that original stays unmodified
            path.append(current_node.create_deepcopy())
        completed_path = self.complete_path(path)
        return completed_path

    def complete_path(self, path):
        # If path is already completed then return that path
        if type(path[-1]) is EndNode:
            return path
        # Otherwise complete the path and then return completed path
        if type(path[-1]) in (Conv2DNode, Pool2DNode):
            path.append(FlattenNode())
        if type(path[-1]) in (FlattenNode, DenseNode, DropoutNode):
            path.append(EndNode())
        return path
