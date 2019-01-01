# Copyright (c) 2019 Edvinas Byla
# Licensed under MIT License

from deepswarm.nodes import InputNode


class Graph:
    def __init__(self):
        self.current_depth = 0
        self.input_node = InputNode()
