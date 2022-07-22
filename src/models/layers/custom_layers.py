import tensorflow.keras as tfk
from tensorflow.keras import layers as tfkl
import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np

def convnet(in_dim : tuple):
    return tfk.Sequential(
    [
        tfk.Input(shape=in_dim),
        tfkl.Conv2D(32, kernel_size=(3, 3), activation="relu",padding = 'same'),
        tfkl.Conv2D(32, kernel_size=(3, 3), activation="relu",padding = 'same'),
        tfkl.MaxPooling2D(pool_size=(2, 2)),
        tfkl.Dropout(0.25),
        tfkl.Conv2D(64, kernel_size=(3, 3), activation="relu",padding = 'same'),
        tfkl.Conv2D(64, kernel_size=(3, 3), activation="relu",padding = 'same'),
        tfkl.MaxPooling2D(pool_size=(2, 2)),
        tfkl.Dropout(0.25),
        tfkl.Flatten(),
        tfkl.Dense(512, activation='relu')
    ]
)

class generator_block(tfkl.Layer):
    def __init__(self, in_dim, scale, n_classes, n, intermediate=None, **kwargs) -> None:
        super(generator_block, self).__init__(name='generator{}'.format(n), **kwargs)
        self.generator = tfkl.Dense(np.product(in_dim[1:]), input_shape=in_dim, activation='relu', name='generator_dense')
        self.intermediate = intermediate(in_dim)
        self.out = tfkl.Dense(n_classes, activation='softmax', name='generator_out')
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_classes' : self.n_classes,
            'in_dim' : self.in_dim,
            'scale' : self.scale,
            'intermediate' : self.intermediate
        })
        return config

    @tf.function
    def call(self, input_tensor, training=False):
        if training:
            x = self.generator(input_tensor)
        if self.intermediate: x = self.intermediate(x)
        return self.out(x)
        
class single_epsilon_generator(tfkl.Layer):
    def __init__(self, in_dim, n_classes, intermediate=None, epsilon=0.05, **kwargs) -> None:
        super(single_epsilon_generator, self).__init__(**kwargs)
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.intermediate = intermediate(in_dim[1:])
        self.out = tfkl.Dense(n_classes, activation='softmax', name='generator_out')
        self.epsilon = tf.repeat(epsilon, repeats=np.product(in_dim))
    
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
        assert len(input_tensor.shape) == 4, "Incorrect shape passed"
        a, b, c, d = input_tensor.shape
        x = tf.reshape(input_tensor, [-1])
        if training: 
            x = self._distr(x)
        if self.intermediate: 
            x = tf.reshape(x, [a, b, c, d])
            x = self.intermediate(x)
        else:
            x = tf.reshape(x, [a, np.product([b, c, d])])
        return self.out(x)
       