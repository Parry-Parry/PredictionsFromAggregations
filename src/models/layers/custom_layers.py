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
    def __init__(self, n_classes, intermediate=None, epsilon=0.05) -> None:
        super(single_epsilon_generator, self).__init__()
        self.intermediate = intermediate
        self.out = tfkl.Dense(n_classes, activation='softmax', name='generator_out')
        self.epsilon = epsilon
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'intermediate': self.intermediate,
            'epsilon': self.epsilon
        })
        return config

    def _distr(self, value):
        sample = np.random.uniform(low=value.numpy()-self.epsilon, high=value.numpy()+self.epsilon)
        x = tf.math.minimum(tf.math.maximum(0.0, sample), 1.0)
        return tf.constant(tf.cast(x, tf.float32), dtype=tf.float32)

    def call(self, input_tensor):
        assert len(input_tensor.shape) == 4, "Incorrect shape passed"
        a, b, c, d = input_tensor.shape
        interim_tensor = tf.reshape(input_tensor, [-1])
        x = tf.map_fn(lambda x : self._distr(x), elems=interim_tensor)
        if self.intermediate: 
            x = tf.reshape(x, [a, b, c, d])
            x = self.intermediate(x)
        else:
            x = tf.reshape(x, [a, np.product([b, c, d])])
        return self.out(x)
       