import tensorflow.keras as tfk 
import tensorflow.keras.layers as tfkl
import tensorflow as tf

from layers.custom_layers import generator_block

def generator_loss(y_true, y_pred, weights):
    cce = tfk.losses.CategoricalCrossentropy()
    return cce(y_true, y_pred) - tf.reduce_sum(weights ** 2) ** 0.5

class stochastic_model(tfk.Model):
    def __init__(self, config, name='') -> None:
        super(self, stochastic_model).__init__(name=name)
        self.generators = [generator_block(config.scale, config.n_classes, i, config.intermediate) for i in range(config.n_gen)]
    def call():
        pass