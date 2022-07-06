import tensorflow.keras as tfk 
import tensorflow.keras.layers as tfkl
import tensorflow as tf

import numpy as np

from src.models.structures import generator_config
from src.models.layers.custom_layers import generator_block

def generator_loss(y_true, y_pred, weights):
    cce = tfk.losses.CategoricalCrossentropy()
    n = len(weights)
    return cce(y_true, y_pred) - tf.reduce_sum([tf.norm(weights[i] - weights[j], ord='fro') for i in range(n) for j in range(n)])

class stochastic_model(tfk.Model):
    def __init__(self, config : generator_config, name='') -> None:
        super(stochastic_model, self).__init__(name=name)
        self.generators = [generator_block(config.in_dim, config.scale, config.n_classes, i, config.intermediate) for i in range(config.n_gen)]
        self.merger = config.merger
        self.out = tfkl.Dense(config.n_classes, activation='softmax')
    def call(self, input_tensor):
        intermediate = tf.concat([gen(input_tensor) for gen in self.generators], axis=0)
        merged = self.merger(intermediate)
        return self.out(merged)

