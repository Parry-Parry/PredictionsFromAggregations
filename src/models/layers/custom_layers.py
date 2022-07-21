import tensorflow.keras as tfk
from tensorflow.keras import layers as tfkl
import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np

class generator_block(tfkl.Layer):
    def __init__(self, in_dim, scale, n_classes, n, intermediate=None) -> None:
        super(generator_block, self).__init__(name='generator{}'.format(n))
        self.dense = tfkl.Dense(128 * scale, input_shape=in_dim, activation='relu', name='generator_dense{}'.format(n))
        self.intermediate = intermediate
        self.out = tfkl.Dense(n_classes, activation='softmax', name='generator_out{}'.format(n))
        
    def call(self, input_tensor):
        x = self.dense(input_tensor)
        if self.intermediate: x = self.intermediate(x)
        return self.out(x)
        
class single_epsilon_generator(tfkl.Layer):
    """
    Generate samples from centroid within its epsilon neighbourhood

    :param int n_generator: The number of samples to be generated from the centroid
    :param float epsilon: Hyperparameter control the size of the neighbourhood
    """
    def __init__(self, in_dim, n_classes, intermediate=None, epsilon=0.05) -> None:
        super(single_epsilon_generator, self).__init__()
        print(np.product(in_dim))
        self.intermediate = intermediate
        self.out = tfkl.Dense(n_classes, activation='softmax', name='generator_out')
        self.epsilon = tf.repeat(epsilon, repeats=np.product(in_dim))
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'epsilon': self.epsilon
        })
        return config

    @tf.function
    def _distr(self, tensor):
        return tfp.distributions.Uniform(low=tensor-self.epsilon, high=tensor+self.epsilon)
    @tf.function
    def call(self, input_tensor):
        assert len(input_tensor.shape) == 4, "Incorrect shape passed"
        a, b, c, d = input_tensor.shape
        interim_tensor = tf.reshape(input_tensor, [-1])
        x = self._distr(interim_tensor)
        if self.intermediate: 
            x = tf.reshape(x, [a, b, c, d])
            x = self.intermediate(x)
        else:
            x = tf.reshape(x, [a, np.product([b, c, d])])
        return self.out(x)
       