# Copyright (c) 2019 Edvinas Byla
# Licensed under MIT License

import context
import tensorflow as tf

from deepswarm.backends import TFKerasBackend
from deepswarm.deepswarm import DeepSwarm

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalize and reshape data
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = x_train.shape[1:]
# Compress data back to tuple
normalized_dataset = (x_train, y_train), (x_test, y_test)
# Create backend responsible for training & validating
backend = TFKerasBackend(
    dataset=normalized_dataset,
    input_shape=input_shape,
    output_size=10
)
# Create DeepSwarm object responsible for optimization
deepswarm = DeepSwarm(backend=backend)
deepswarm.find_topology(max_depth=2, swarm_size=2)
