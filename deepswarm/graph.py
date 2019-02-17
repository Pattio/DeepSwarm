# Copyright (c) 2019 Edvinas Byla
# Licensed under MIT License

from . import cfg
from .log import Log
from .nodes import (Conv2DNode, DenseNode, DropoutNode, EndNode, FlattenNode,
    InputNode, NeighbourNode, Pool2DNode)


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
            for available_transition in available_transitions:
                neighbour_node = self.get_node(available_transition(), depth + 1)
                node.neighbours.append(NeighbourNode(neighbour_node, cfg['pheromone']['start']))
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
        if type(path[-1]) in (Conv2DNode, Pool2DNode):
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
                for item in info:
                    Log.info(item)
        Log.header("PHEROMONE END", type="RED")
