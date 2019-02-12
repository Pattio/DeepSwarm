# Copyright (c) 2019 Edvinas Byla
# Licensed under MIT License

import copy
import random
from . import config as cfg


class NodeAttribute:
    def __init__(self, name, options):
        self.name = name
        self.dict = {option: cfg.pheromone['start'] for option in options}


class NeighbourNode:
    def __init__(self, node, pheromone):
        self.node = node
        self.pheromone = pheromone


class Node:
    def __init__(self, name):
        self.name = name
        self.is_expanded = False
        self.attributes = []
        self.neighbours = []
        self.available_transitions = []

    def select_attributes(self, custom_select):
        selected_attributes = {}
        for attribute in self.attributes:
            value = custom_select(attribute.dict)
            selected_attributes[attribute.name] = value
        # For each selected attribute create class attribute
        for key, value in selected_attributes.items():
            setattr(self, key, value)

    def select_custom_attributes(self, custom_select):
        def select(dict):
            return custom_select(list(dict.items()))[0]
        self.select_attributes(select)

    def select_random_attributes(self):
        def select(dict):
            return random.choice(list(dict.keys()))
        self.select_attributes(select)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            # Skip unnecessary stuff to make copying more efficient
            if k in ["neighbours", "available_transitions"]:
                v = []
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    def create_deepcopy(self):
        return copy.deepcopy(self)


class InputNode(Node):
    def __init__(self):
        super().__init__("InputNode")
        self.available_transitions = [Conv2DNode]


class Conv2DNode(Node):
    def __init__(self):
        super().__init__("Conv2DNode")
        self.attributes = [
            NodeAttribute("filter_number", [16, 32, 64]),
            NodeAttribute("kernel_size", [1, 3, 5]),
            NodeAttribute("activation", ["ReLU"]),
        ]
        self.available_transitions = [Conv2DNode, Pool2DNode, FlattenNode]


class Pool2DNode(Node):
    def __init__(self):
        super().__init__("Pool2DNode")
        self.attributes = [
            NodeAttribute("type", ["max"]),
            NodeAttribute("pool_size", [2]),
            NodeAttribute("stride", [2, 3]),
        ]
        self.available_transitions = [Conv2DNode, FlattenNode]


class FlattenNode(Node):
    def __init__(self):
        super().__init__("FlattenNode")
        self.available_transitions = [DenseNode]


class DenseNode(Node):
    def __init__(self):
        super().__init__("DenseNode")
        self.attributes = [
            NodeAttribute("output_size", [128, 256, 512]),
            NodeAttribute("activation", ["ReLU"]),
        ]
        self.available_transitions = [DenseNode, DropoutNode]


class DropoutNode(Node):
    def __init__(self):
        super().__init__("DropoutNode")
        self.attributes = [
            NodeAttribute("rate", [0.1, 0.3, 0.5]),
        ]
        self.available_transitions = [DenseNode]


class EndNode(Node):
    def __init__(self):
        super().__init__("EndNode")
