# Copyright (c) 2019 Edvinas Byla
# Licensed under MIT License

import context
from deepswarm.network_graph import Graph

graph = Graph()
input_node = graph.input_node
input_node.expand()
input_node.neighbours[0].expand()
print(input_node.neighbours[0].neighbours)
