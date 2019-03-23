# Copyright (c) 2019 Edvinas Byla
# Licensed under MIT License

import time
from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow.keras import backend as K
from . import cfg


class Dataset:
    def __init__(self, training_examples, training_labels, testing_examples, testing_labels,
     validation_data=None, validation_split=0.1):
        self.x_train = training_examples
        self.y_train = training_labels
        self.x_test = testing_examples
        self.y_test = testing_labels
        self.validation_data = validation_data
        self.validation_split = validation_split


class BaseBackend(ABC):
    def __init__(self, dataset, optimizer=None):
        self.dataset = dataset
        self.optimizer = optimizer

    @abstractmethod
    def generate_model(self, path):
        """Create and return a backend model representation.

        Args:
            path [Node]: list of nodes where each node represents single
            network layer, path starts with InputNode and ends with EndNode
        Returns:
            model which represents neural network structure in the implemented
            backend, this model can be evaluated using evaluate_model method

        """

    @abstractmethod
    def reuse_model(self, old_model, new_model_path, distance):
        """Create new model, by reusing layers (and their weights) from old model.

        Args:
            old_model: old model which represents neural network structure
            new_model_path [Node]: path representing new model
            distance (int): distance which shows how many layers from old model need
            to be removed in order to create a base for new model i.e. if old model is
            NodeA->NodeB->NodeC->NodeD and new model is NodeA->NodeB->NodeC->NodeE, distance = 1
        Returns:
            model which represents neural network structure

        """

    @abstractmethod
    def train_model(self, model):
        """Train model which was created using generate_model method.

        Args:
            model: model which represents neural network structure
        Returns:
            model

        """

    @abstractmethod
    def evaluate_model(self, model):
        """Evaluate model which was created using generate_model method.

        Args:
            model: model which represents neural network structure
        Returns:
            loss & accuracy tuple

        """

    @abstractmethod
    def save_model(self, model, path):
        """Saves model on disk

        Args:
            model: model which represents neural network structure
            path: string which represents model location
        """

    @abstractmethod
    def load_model(self, path):
        """Load model from disk, in case of fail should return None

        Args:
            path: string which represents model location
        Returns:
            model: model which represents neural network structure, or in case
            fail None
        """

    @abstractmethod
    def free_gpu(self):
        """ Frees gpu memory
        """


class TFKerasBackend(BaseBackend):
    def __init__(self, dataset, optimizer=None):
        super().__init__(dataset, optimizer)
        self.data_format = K.image_data_format()

    def generate_model(self, path):
        # Create input layer
        input_layer = self.create_layer(path[0])
        layer = input_layer
        # Convert each node to layer and then connect it to the previous layer
        for node in path[1:]:
            layer = self.create_layer(node)(layer)
        # Return generated model
        return tf.keras.Model(inputs=input_layer, outputs=layer)

    def reuse_model(self, old_model, new_model_path, distance):
        # Find starting point of new model
        starting_point = len(new_model_path) - distance
        last_layer = old_model.layers[starting_point - 1].output
        # Append layers from new model to the old model
        for node in new_model_path[starting_point:]:
            last_layer = self.create_layer(node)(last_layer)
        # Return new model
        return tf.keras.Model(inputs=old_model.inputs, outputs=last_layer)

    def create_layer(self, node):
        # Workaround to prevent Keras from throwing an exception ("All layer names should be unique.")
        # It happens when new layers are appended to an existing model, but Keras fails to increment
        # repeating layer names i.e. conv_1 -> conv_2
        parameters = {'name': str(time.time())}

        if node.type == 'Input':
            parameters['shape'] = node.shape
            return tf.keras.Input(**parameters)

        if node.type == 'Conv2D':
            parameters.update({
                'filters': node.filter_count,
                'kernel_size': node.kernel_size,
                'padding': 'same',
                'data_format': self.data_format,
                'activation': self.map_activation(node.activation),
            })
            return tf.keras.layers.Conv2D(**parameters)

        if node.type == 'Pool2D':
            parameters.update({
                'pool_size': node.pool_size,
                'strides': node.stride,
                'padding': 'same',
                'data_format': self.data_format,
            })
            if node.pool_type == 'max':
                return tf.keras.layers.MaxPooling2D(**parameters)
            elif node.pool_type == 'average':
                return tf.keras.layers.AveragePooling2D(**parameters)

        if node.type == 'BatchNormalization':
            return tf.keras.layers.BatchNormalization(**parameters)

        if node.type == 'Flatten':
            return tf.keras.layers.Flatten(**parameters)

        if node.type == 'Dense':
            parameters.update({
                'units': node.output_size,
                'activation': self.map_activation(node.activation),
            })
            return tf.keras.layers.Dense(**parameters)

        if node.type == 'Dropout':
            parameters.update({
                'rate': node.rate,
            })
            return tf.keras.layers.Dropout(**parameters)

        if node.type == 'Output':
            parameters.update({
                'units': node.output_size,
                'activation': self.map_activation(node.activation),
            })
            return tf.keras.layers.Dense(**parameters)

        raise Exception('Not handled node type: %s' % str(node))

    def map_activation(self, activation):
        if activation == "ReLU":
            return tf.keras.activations.relu
        if activation == "ELU":
            return tf.keras.activations.elu
        if activation == "LeakyReLU":
            return tf.nn.leaky_relu
        if activation == "Sigmoid":
            return tf.keras.activations.sigmoid
        if activation == "Softmax":
            return tf.keras.activations.softmax
        raise Exception('Not handled activation: %s' % str(activation))

    def train_model(self, model):
        optimizer_parameters = {
            'optimizer': 'adam',
            'loss': cfg['backend']['loss'],
            'metrics': ['accuracy'],
        }

        # If user specified custom optimizer, use it instead of the default one
        if self.optimizer is not None:
            optimizer_parameters['optimizer'] = self.optimizer

        model.compile(**optimizer_parameters)

        early_stop_parameters = {
            'patience': cfg['backend']['patience'],
            'verbose': 1,
            'restore_best_weights': True,
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
        self.free_gpu()

    def load_model(self, path):
        try:
            model = tf.keras.models.load_model(path)
            return model
        except:
            return None

    def free_gpu(self):
        K.clear_session()
