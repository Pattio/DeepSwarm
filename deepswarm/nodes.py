# Copyright (c) 2019 Edvinas Byla
# Licensed under MIT License

import copy


class Node:
    def __init__(self, name):
        self.name = name
        self.neighbours = []
        self.is_expanded = False

    @classmethod
    def available_instances(cls):
        return []

    @staticmethod
    def available_transitions():
        return []

    def expand(self):
        # Expand node only if it has not been expanded before
        if self.is_expanded:
            return
        else:
            self.is_expanded = True

        for node in self.available_transitions():
            self.neighbours += node.available_instances()

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            # Skip neighbours to make copying more efficient
            if k == "neighbours":
                v = []
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    def create_deepcopy(self):
        return copy.deepcopy(self)


class InputNode(Node):
    def __init__(self):
        super().__init__("InputNode")

    @staticmethod
    def available_transitions():
        return [
            Conv2DNode,
        ]


class Conv2DNode(Node):
    def __init__(self, kernel_size):
        super().__init__("Conv2DNode-%d" % kernel_size)
        self.kernel_size = kernel_size

    @classmethod
    def available_instances(cls):
        return [
            cls(1),
            cls(3),
            cls(5),
        ]

    @staticmethod
    def available_transitions():
        return [
            Conv2DNode,
            MaxPool2DNode,
            FlattenNode,
        ]


class MaxPool2DNode(Node):
    def __init__(self, pool_size, strides):
        super().__init__("MaxPool2DNode-%d-%d" % (pool_size, strides))
        self.pool_size = pool_size
        self.strides = strides

    @classmethod
    def available_instances(cls):
        return [
            cls(2, 2),
            cls(2, 3),
        ]

    @staticmethod
    def available_transitions():
        return [
            Conv2DNode,
            FlattenNode,
        ]


class FlattenNode(Node):
    def __init__(self):
        super().__init__("FlattenNode")

    @classmethod
    def available_instances(cls):
        return [
            cls(),
        ]

    @staticmethod
    def available_transitions():
        return [
            DenseNode,
        ]


class DenseNode(Node):
    def __init__(self, output_size, activation):
        super().__init__("DenseNode-%d-%s" % (output_size, activation))
        self.output_size = output_size
        self.activation = activation

    @classmethod
    def available_instances(cls):
        return [
            cls(128, "ReLu"),
            cls(256, "ReLu"),
            cls(512, "ReLu"),
        ]

    @staticmethod
    def available_transitions():
        return [
            DenseNode,
            DropoutNode,
        ]


class DropoutNode(Node):
    def __init__(self, rate):
        super().__init__("DropoutNode-%f" % rate)
        self.rate = rate

    @classmethod
    def available_instances(cls):
        return [
            cls(0.1),
            cls(0.2),
            cls(0.3),
            cls(0.4),
        ]

    @staticmethod
    def available_transitions():
        return [
            DenseNode,
        ]


class EndNode(Node):
    def __init__(self):
        super().__init__("EndNode")
