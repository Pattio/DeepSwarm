# Copyright (c) 2019 Edvinas Byla
# Licensed under MIT License

import tensorflow as tf
from tensorflow.keras import backend as K
from .nodes import (Conv2DNode, DenseNode, DropoutNode, EndNode, FlattenNode, InputNode, Pool2DNode)


class BaseBackend:
    def __init__(self, dataset, input_shape, output_size):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = dataset
        self.input_shape = input_shape
        self.output_size = output_size

    def generate_model(self, path):
        """Create and return a backend model representation.

        Args:
            path (Node): list of nodes where each node represents single
            network layer, path starts with InputNode and ends with EndNode
        Returns:
            model which represents neural network structure in the implemented
            backend, this model can be evaluated using evaluate_model method

        """
        raise NotImplementedError()

    def train_model(self, model):
        """Train model which was created using generate_model method.

        Args:
            model: model which represents neural network structure
        Returns:
            model

        """
        raise NotImplementedError()

    def evaluate_model(self, model):
        """Evaluate model which was created using generate_model method.

        Args:
            model: model which represents neural network structure
        Returns:
            loss & accuracy tuple

        """
        raise NotImplementedError()

    def save_model(self, model, path):
        """Saves model on disk

        Args:
            model: model which represents neural network structure
            path: string which represents model location
        """
        raise NotImplementedError()

    def load_model(self, path):
        """Load model from disk, in case of fail should return None

        Args:
            path: string which represents model location
        Returns:
            model: model which represents neural network structure, or in case
            fail None
        """
        raise NotImplementedError()


class TFKerasBackend(BaseBackend):
    def __init__(self, dataset, input_shape, output_size):
        super().__init__(dataset, input_shape, output_size)

    def generate_model(self, path):
        data_format = K.image_data_format()
        model = tf.keras.models.Sequential()

        for idx, node in enumerate(path):
            if type(node) is Conv2DNode:
                # Set input shape only for first layer after input
                if idx > 0 and type(path[idx - 1]) is InputNode:
                    model.add(
                        tf.keras.layers.Conv2D(
                            filters=2,
                            kernel_size=node.kernel_size,
                            padding='same',
                            data_format=data_format,
                            activation=tf.nn.relu,
                            input_shape=self.input_shape,
                        )
                    )
                else:
                    model.add(
                        tf.keras.layers.Conv2D(
                            filters=2,
                            kernel_size=node.kernel_size,
                            padding='same',
                            data_format=data_format,
                            activation=tf.nn.relu,
                        )
                    )
            elif type(node) is Pool2DNode:
                # TODO: add support for average
                model.add(
                    tf.keras.layers.MaxPooling2D(
                        pool_size=node.pool_size,
                        strides=node.stride,
                        padding='same',
                        data_format=data_format,
                    )
                )
            elif type(node) is FlattenNode:
                model.add(
                    tf.keras.layers.Flatten()
                )
            elif type(node) is DenseNode:
                model.add(
                    tf.keras.layers.Dense(
                        units=node.output_size,
                        activation=tf.nn.relu,
                    )
                )
            elif type(node) is DropoutNode:
                model.add(
                    tf.keras.layers.Dropout(node.rate)
                )
            elif type(node) is EndNode:
                model.add(
                    tf.keras.layers.Dense(
                        units=self.output_size,
                        activation=tf.nn.softmax
                    )
                )
        return model

    def train_model(self, model):
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        model.fit(self.x_train, self.y_train, epochs=1, batch_size=1000)
        return model

    def evaluate_model(self, model):
        loss, accuracy = model.evaluate(self.x_test, self.y_test)
        return (loss, accuracy)

    def save_model(self, model, path):
        model.save(path)

    def load_model(self, path):
        try:
            model = tf.keras.models.load_model(path)
            return model
        except:
            return None
