# Copyright (c) 2019 Edvinas Byla
# Licensed under MIT License

import random
from .nodes import (Conv2DNode, DenseNode, DropoutNode, EndNode, FlattenNode,
                    InputNode, MaxPool2DNode)


class Graph:
    def __init__(self):
        self.allowed_depth = 1
        self.input_node = InputNode()

    def generate_random_path(self):
        path = [self.input_node.create_deepcopy()]
        current_node = self.input_node
        for _ in range(self.allowed_depth):
            current_node.expand()
            # Select new random node and append it to the path
            random_index = random.randint(0, len(current_node.neighbours) - 1)
            current_node = current_node.neighbours[random_index]
            # Add only copy of the node, so that original stays unmodified
            path.append(current_node.create_deepcopy())
        completed_path = self.complete_path(path)
        return completed_path

    def complete_path(self, path):
        # If path is already completed then return that path
        if type(path[-1]) is EndNode:
            return path
        # Otherwise complete the path and then return completed path
        if type(path[-1]) in (Conv2DNode, MaxPool2DNode):
            path.append(FlattenNode())
        if type(path[-1]) in (FlattenNode, DenseNode, DropoutNode):
            path.append(EndNode())
        return path
