# Copyright (c) 2019 Edvinas Byla
# Licensed under MIT License

import copy
import random

from . import cfg, nodes


class NodeAttribute:
    """Class responsible for encapsulating Node's attribute."""

    def __init__(self, name, options):
        self.name = name
        self.dict = {option: cfg['aco']['pheromone']['start'] for option in options}


class NeighbourNode:
    """Class responsible for encapsulating Node's neighbour."""

    def __init__(self, node, heuristic, pheromone=cfg['aco']['pheromone']['start']):
        self.node = node
        self.heuristic = heuristic
        self.pheromone = pheromone


class Node:
    """Class responsible for representing Node."""

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
        """Create Node's instance using given type.

        Args:
            type (str): type defined in .yaml file.
        Returns:
            Node's instance.
        """

        for node in nodes:
            if nodes[node]['type'] == type:
                return cls(node)
        raise Exception('Type does not exist: %s' % str(type))

    def setup_attributes(self):
        """Adds attributes from the settings file."""

        self.attributes = []
        for attribute_name in nodes[self.name]['attributes']:
            attribute_value = nodes[self.name]['attributes'][attribute_name]
            self.attributes.append(NodeAttribute(attribute_name, attribute_value))

    def setup_transitions(self):
        """Adds transitions from the settings file."""

        self.available_transitions = []
        for transition_name in nodes[self.name]['transitions']:
            heuristic_value = nodes[self.name]['transitions'][transition_name]
            self.available_transitions.append((transition_name, heuristic_value))

    def select_attributes(self, custom_select):
        """Selects attributes using a given select rule.

        Args:
            custom_select: select function which takes dictionary containing
            (attribute, value) pairs and returns selected value.
        """

        selected_attributes = {}
        for attribute in self.attributes:
            value = custom_select(attribute.dict)
            selected_attributes[attribute.name] = value

        # For each selected attribute create class attribute
        for key, value in selected_attributes.items():
            setattr(self, key, value)

    def select_custom_attributes(self, custom_select):
        """Wraps select_attributes method by converting the attribute dictionary
        to list of tuples (attribute_value, pheromone, heuristic).

        Args:
            custom_select: selection function which takes a list of tuples
            containing (attribute_value, pheromone, heuristic).
        """

        # Define a function which transforms attributes before selecting them
        def select_transformed_custom_attributes(attribute_dictionary):
            # Convert to list of tuples containing (attribute_value, pheromone, heuristic)
            values = [(value, pheromone, 1.0) for value, pheromone in attribute_dictionary.items()]
            # Return value, which was selected using custom select
            return custom_select(values)
        self.select_attributes(select_transformed_custom_attributes)

    def select_random_attributes(self):
        """Selects random attributes."""

        self.select_attributes(lambda dict: random.choice(list(dict.keys())))

    def create_deepcopy(self):
        """Returns a newly created copy of Node object."""

        return copy.deepcopy(self)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            # Skip unnecessary stuff in order to make copying more efficient
            if k in ["neighbours", "available_transitions"]:
                v = []
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    def __str__(self):
        attributes = ', '.join([a.name + ":" + str(getattr(self, a.name)) for a in self.attributes])
        return self.name + "(" + attributes + ")"
