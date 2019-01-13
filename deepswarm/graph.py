# Copyright (c) 2019 Edvinas Byla
# Licensed under MIT License

import random
from .nodes import (Conv2DNode, DenseNode, DropoutNode, EndNode, FlattenNode, InputNode, MaxPool2DNode)


class Graph:
    def __init__(self, current_depth=1):
        self.current_depth = current_depth
        self.input_node = InputNode()

    def increase_depth(self):
        self.current_depth += 1

    def generate_random_path(self):
        path = [self.input_node.create_deepcopy()]
        current_node = self.input_node
        for _ in range(self.current_depth):
            current_node.expand()
            # Stop expanding node, when it can't be expanded any more
            if not current_node.neighbours:
                break
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
