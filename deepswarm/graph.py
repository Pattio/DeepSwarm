# Copyright (c) 2019 Edvinas Byla
# Licensed under MIT License

import random
from .nodes import InputNode


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
        return path
