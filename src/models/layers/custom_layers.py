import tensorflow.keras as tfk
import tensorflow.math as tfm
from tensorflow.keras import layers as tfkl
import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np

def convnet(in_dim):
    return tfk.Sequential(
    [
        tfkl.Conv2D(32, kernel_size=(3, 3), activation="relu",padding='same'),
        tfkl.Conv2D(32, kernel_size=(3, 3), activation="relu",padding='same'),
        tfkl.MaxPooling2D(pool_size=(2, 2)),
        tfkl.Dropout(0.25),
        tfkl.Conv2D(64, kernel_size=(3, 3), activation="relu",padding='same'),
        tfkl.Conv2D(64, kernel_size=(3, 3), activation="relu",padding='same'),
        tfkl.MaxPooling2D(pool_size=(2, 2)),
        tfkl.Dropout(0.25),
        tfkl.Flatten(),
        tfkl.Dense(512, activation='relu')
    ]
)

class generator_block(tfkl.Layer):
    def __init__(self, in_dim, scale, n_classes, n, intermediate=None, **kwargs) -> None:
        super(generator_block, self).__init__(name='generator{}'.format(n), **kwargs)
        self.in_dim = in_dim
        self.generator = tfkl.Dense(tfm.reduce_prod(in_dim[1:]), activation='relu', name='generator_dense')
        self.intermediate = intermediate(in_dim[1:])
        self.out = tfkl.Dense(n_classes, activation='softmax', name='generator_out')
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'in_dim' : self.in_dim,
            'intermediate' : self.intermediate
        })
        return config

    #@tf.function
    def call(self, input_tensor, training=False):
        x = input_tensor
        if training:
            x = tf.reshape(x, (self.in_dim[0], tfm.reduce_prod(self.in_dim[1:])))
            x = self.generator(x)
            x = tf.reshape(x, self.in_dim)
        if self.intermediate: x = self.intermediate(x)
        return self.out(x)
        
class single_epsilon_generator(tfkl.Layer):
    def __init__(self, in_dim, n_classes, intermediate=None, epsilon=0.05, **kwargs) -> None:
        super(single_epsilon_generator, self).__init__(**kwargs)
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.intermediate = intermediate(in_dim[1:])
        self.out = tfkl.Dense(n_classes, activation='softmax', name='generator_out')
        self.epsilon = tf.repeat(epsilon, repeats=tfm.reduce_prod(in_dim))
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_classes' : self.n_classes,
            'in_dim' : self.in_dim,
            'epsilon': self.epsilon,
            'intermediate' : self.intermediate
        })
        return config

    def _distr(self, tensor):
        return tfp.distributions.Uniform(low=tensor-self.epsilon, high=tensor+self.epsilon).sample()
        
    @tf.function
    def call(self, input_tensor, training=False):
        shape = input_tensor.shape
        assert len(shape) == 4, "Incorrect shape passed"
        x = tf.reshape(input_tensor, [-1])
        if training: 
            x = self._distr(x)
        if self.intermediate: 
            x = tf.reshape(x, shape)
            x = self.intermediate(x)
        else:
            x = tf.reshape(x, [shape[0], tfm.reduce_sum(shape[1:])])
        return self.out(x)
       