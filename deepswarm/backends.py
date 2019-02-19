# Copyright (c) 2019 Edvinas Byla
# Licensed under MIT License

import tensorflow as tf
from tensorflow.keras import backend as K
from . import cfg
from .nodes import (Conv2DNode, DenseNode, DropoutNode, EndNode, FlattenNode, InputNode, Pool2DNode)


class Dataset:
    def __init__(self, training_examples, training_labels, testing_examples, testing_labels,
     validation_data=None, validation_split=0.1):
        self.x_train = training_examples
        self.y_train = training_labels
        self.x_test = testing_examples
        self.y_test = testing_labels
        self.validation_data = validation_data
        self.validation_split = validation_split


class BaseBackend:
    def __init__(self, dataset, input_shape, output_size):
        self.dataset = dataset
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
                conv2d_parameters = {
                    'filters': node.filter_number,
                    'kernel_size': node.kernel_size,
                    'padding': 'same',
                    'data_format': data_format,
                    'activation': tf.nn.relu,
                }
                # Set input shape only for first layer after input
                if idx > 0 and type(path[idx - 1]) is InputNode:
                    conv2d_parameters['input_shape'] = self.input_shape

                model.add(tf.keras.layers.Conv2D(**conv2d_parameters))
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

        early_stop_parameters = {
            'patience': cfg['backend']['patience'],
            'verbose': 1
        }
        # Set user defined metrics
        if cfg['metrics'] == 'loss':
            early_stop_parameters['monitor'] = 'val_loss'
        else:
            early_stop_parameters['monitor'] = 'val_acc'

        early_stop_callback = tf.keras.callbacks.EarlyStopping(**early_stop_parameters)

        # Setup training parameters
        fit_parameters = {
            'x': self.dataset.x_train,
            'y': self.dataset.y_train,
            'epochs': cfg['backend']['epochs'],
            'batch_size': cfg['backend']['batch_size'],
            'callbacks': [early_stop_callback],
        }

        # If no validation data was provided use validation split
        if self.dataset.validation_data is None:
            fit_parameters['validation_split'] = self.dataset.validation_split
        # Othwerwise use provided validation data
        else:
            fit_parameters['validation_data'] = self.dataset.validation_data
        # Train and return model
        model.fit(**fit_parameters)
        return model

    def evaluate_model(self, model):
        loss, accuracy = model.evaluate(self.dataset.x_test, self.dataset.y_test)
        return (loss, accuracy)

    def save_model(self, model, path):
        model.save(path)

    def load_model(self, path):
        try:
            model = tf.keras.models.load_model(path)
            return model
        except:
            return None
