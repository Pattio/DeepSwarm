# Copyright (c) 2019 Edvinas Byla
# Licensed under MIT License

import time
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
    def __init__(self, dataset):
        self.dataset = dataset

    def generate_model(self, path):
        """Create and return a backend model representation.

        Args:
            path [Node]: list of nodes where each node represents single
            network layer, path starts with InputNode and ends with EndNode
        Returns:
            model which represents neural network structure in the implemented
            backend, this model can be evaluated using evaluate_model method

        """
        raise NotImplementedError()

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

    def free_gpu(self):
        """ Frees gpu memory
        """
        raise NotImplementedError()


class TFKerasBackend(BaseBackend):
    def __init__(self, dataset):
        super().__init__(dataset)
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

        if type(node) is InputNode:
            parameters['shape'] = node.shape
            return tf.keras.Input(**parameters)

        if type(node) is Conv2DNode:
            parameters.update({
                'filters': node.filter_number,
                'kernel_size': node.kernel_size,
                'padding': 'same',
                'data_format': self.data_format,
                'activation': tf.nn.relu,
            })
            return tf.keras.layers.Conv2D(**parameters)

        if type(node) is Pool2DNode:
            parameters.update({
                'pool_size': node.pool_size,
                'strides': node.stride,
                'padding': 'same',
                'data_format': self.data_format,
            })
            if node.type == 'max':
                return tf.keras.layers.MaxPooling2D(**parameters)
            elif node.type == 'average':
                return tf.keras.layers.AveragePooling2D(**parameters)

        if type(node) is FlattenNode:
            return tf.keras.layers.Flatten(**parameters)

        if type(node) is DenseNode:
            parameters.update({
                'units': node.output_size,
                'activation': tf.nn.relu,
            })
            return tf.keras.layers.Dense(**parameters)

        if type(node) is DropoutNode:
            parameters.update({
                'rate': node.rate,
            })
            return tf.keras.layers.Dropout(**parameters)

        if type(node) is EndNode:
            parameters.update({
                'units': node.output_size,
                'activation': tf.nn.softmax,
            })
            return tf.keras.layers.Dense(**parameters)

        raise Exception('Not handled node type: %s' % str(node))

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
        self.free_gpu()

    def load_model(self, path):
        try:
            model = tf.keras.models.load_model(path)
            return model
        except:
            return None

    def free_gpu(self):
        K.clear_session()
