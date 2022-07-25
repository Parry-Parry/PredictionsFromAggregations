import tensorflow.keras as tfk 
import tensorflow.keras.layers as tfkl
import tensorflow.math as tfm
import tensorflow as tf

import numpy as np

from src.models.structures import generator_config
from src.models.layers.custom_layers import generator_block

def distance_loss(y_true, weights, interim_preds):
    cce = tfk.losses.CategoricalCrossentropy()
    n = len(weights)
    weight_norm = tf.reduce_sum([tf.norm(weights[i] - weights[j], ord='fro') for i in range(n) for j in range(n)], name="Norm of Weight Diff")
    cce_sum = tfm.reduce_sum(tf.map_fn(lambda x : cce(y_true, x), elems=interim_preds), axis=0, name="Sum of CE over Generated Preds")

    return cce_sum - weight_norm

def ensemble_loss(y_true, interim_preds):
    cce = tfk.losses.CategoricalCrossentropy()

    return tfm.reduce_sum(tf.map_fn(lambda x : cce(y_true, x), elems=interim_preds), axis=0, name="Sum of CE over Generated Preds")

class generator_model(tfk.Model):
    def __init__(self, config : generator_config, name='Generator Stack') -> None:
        super(generator_model, self).__init__(name=name)
        self.generators = [generator_block(config.in_dim, config.scale, config.n_classes, i, config.intermediate) for i in range(config.n_gen)]

    def _max_proba(self, proba):
        totals = tfm.reduce_sum(proba, axis=0)

        return tf.one_hot(tf.argmax(totals), depth=1, on_value=1, off_value=0)

    def call(self, input_tensor, training=False):
        intermediate = tf.stack([gen(input_tensor, training) for gen in self.generators], axis=0)
        if training: return self._max_proba(intermediate), intermediate 
        
        return self._max_proba(intermediate)

