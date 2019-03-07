# Copyright (c) 2019 Edvinas Byla
# Licensed under MIT License

import copy
import random
from . import cfg, nodes


class NodeAttribute:
    def __init__(self, name, options):
        self.name = name
        self.dict = {option: cfg['aco']['pheromone']['start'] for option in options}


class NeighbourNode:
    def __init__(self, node, heuristic, pheromone=cfg['aco']['pheromone']['start']):
        self.node = node
        self.heuristic = heuristic
        self.pheromone = pheromone


class Node:
    def __init__(self, name):
        self.name = name
        self.neighbours = []
        self.is_expanded = False
        self.type = nodes[self.name]['type']
        self.setup_attributes()
        self.setup_transitions()
        self.select_random_attributes()

    @classmethod
    def create_using_type(cls, type):
        """Create node instance using given type.

        Args:
            type (str): type defined in .yaml file
        Returns:
            Node instance

        """
        for node in nodes:
            if nodes[node]['type'] == type:
                return cls(node)
        raise Exception('Type does not exist: %s' % str(type))

    def setup_attributes(self):
        self.attributes = []
        for attribute_name in nodes[self.name]['attributes']:
            attribute_value = nodes[self.name]['attributes'][attribute_name]
            self.attributes.append(NodeAttribute(attribute_name, attribute_value))

    def setup_transitions(self):
        self.available_transitions = []
        for transition_name in nodes[self.name]['transitions']:
            heuristic_value = nodes[self.name]['transitions'][transition_name]
            self.available_transitions.append((transition_name, heuristic_value))

    def select_attributes(self, custom_select):
        selected_attributes = {}
        for attribute in self.attributes:
            value = custom_select(attribute.dict)
            selected_attributes[attribute.name] = value
        # For each selected attribute create class attribute
        for key, value in selected_attributes.items():
            setattr(self, key, value)

    def select_custom_attributes(self, custom_select):
        # Define function which transforms attributes, before selecting them
        def select_transformed_custom_attributes(attribute_dictionary):
            # Convert to list of tuples containing (attribute_value, pheromone, heuristic)
            values = [(value, pheromone, 1.0) for value, pheromone in attribute_dictionary.items()]
            # Return value, which was selected using custom select
            return custom_select(values)
        self.select_attributes(select_transformed_custom_attributes)

    def select_random_attributes(self):
        self.select_attributes(lambda dict: random.choice(list(dict.keys())))

    def __str__(self):
        attributes = ', '.join([a.name + ":" + str(getattr(self, a.name)) for a in self.attributes])
        return self.name + "(" + attributes + ")"

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
