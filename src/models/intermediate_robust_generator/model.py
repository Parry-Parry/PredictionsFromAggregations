import tensorflow.keras as tfk 
import tensorflow.keras.layers as tfkl
import tensorflow.math as tfm
import tensorflow as tf

import numpy as np

from src.models.structures import generator_config
from src.models.layers.custom_layers import generator_block

class generator_model(tfk.Model):
    def __init__(self, config : generator_config, name='Generator Stack') -> None:
        super(generator_model, self).__init__(name=name)
        self.generators = [generator_block(config.in_dim, config.scale, config.n_classes, i, config.intermediate) for i in range(config.n_gen)]

    def _max_proba(self, proba):
        totals = tfm.reduce_sum(proba, axis=0)
        probs = tf.map_fn(lambda x : tf.one_hot(tf.argmax(x), depth=x.shape[0], on_value=1, off_value=0, dtype=tf.float32), elems=totals)

        #probs = tf.reduce_mean(proba, axis=0)

        return probs

    def call(self, input_tensor, training=False):
        intermediate = tf.stack([gen(input_tensor, training) for gen in self.generators], axis=0)
        if training: return self._max_proba(intermediate), intermediate 

        return self._max_proba(intermediate)

