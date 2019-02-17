# Copyright (c) 2019 Edvinas Byla
# Licensed under MIT License

import copy
import importlib
import random
from . import cfg, nodes


class NodeAttribute:
    def __init__(self, name, options):
        self.name = name
        self.dict = {option: cfg['pheromone']['start'] for option in options}


class NeighbourNode:
    def __init__(self, node, pheromone):
        self.node = node
        self.pheromone = pheromone


class Node:
    def __init__(self, name):
        self.name = name
        self.is_expanded = False
        self.neighbours = []
        self.setup_attributes()
        self.setup_transitions()

    def setup_attributes(self):
        self.attributes = []
        for attribute_name in nodes[self.name]['attributes']:
            attribute_value = nodes[self.name]['attributes'][attribute_name]
            self.attributes.append(NodeAttribute(attribute_name, attribute_value))

    def setup_transitions(self):
        self.available_transitions = []
        module = importlib.import_module('deepswarm.nodes')
        for transition_name in nodes[self.name]['transitions']:
            transition_class = getattr(module, transition_name)
            self.available_transitions.append(transition_class)

    def select_attributes(self, custom_select):
        selected_attributes = {}
        for attribute in self.attributes:
            value = custom_select(attribute.dict)
            selected_attributes[attribute.name] = value
        # For each selected attribute create class attribute
        for key, value in selected_attributes.items():
            setattr(self, key, value)

    def select_custom_attributes(self, custom_select):
        self.select_attributes(lambda dict: custom_select(list(dict.items()))[0])

    def select_random_attributes(self):
        self.select_attributes(lambda dict: random.choice(list(dict.keys())))

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


class Conv2DNode(Node):
    def __init__(self):
        super().__init__("Conv2DNode")


class Pool2DNode(Node):
    def __init__(self):
        super().__init__("Pool2DNode")


class FlattenNode(Node):
    def __init__(self):
        super().__init__("FlattenNode")


class DenseNode(Node):
    def __init__(self):
        super().__init__("DenseNode")


class DropoutNode(Node):
    def __init__(self):
        super().__init__("DropoutNode")


class EndNode(Node):
    def __init__(self):
        super().__init__("EndNode")
