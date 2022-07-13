import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
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
    def __init__(self,  in_dim, scale, n_classes, intermediate=None, epsilon=0.05) -> None:
        super(single_epsilon_generator, self).__init__()
        self.intermediate = intermediate
        self.out = tfkl.Dense(n_classes, activation='softmax', name='generator_out')
        self.epsilon = epsilon

    def _distr(self, value):
        value = value.numpy()
        distr = tfp.distributions.uniform(low=value-self.epsilon, high=value+self.epsilon)
        x = tf.math.minimum(tf.math.maximum(0.0, distr.sample()), 1.0)
        return tf.constant(x, dtype=tf.float32)

    def _sample(self, input_tensor):
        x = [self._distr(x) for x in tf.unstack(input_tensor)]
        #x = tf.map_fn(self._distr, elems=tensor)
        return tf.concat(x, axis=0)

    def call(self, input_tensor):
        #x = tf.map_fn(self._sample, elems=input_tensor)
        x = tf.concat([self._sample(x) for x in tf.unstack(input_tensor)], axis=0)
        if self.intermediate: x = self.intermediate(x)
        return self.out(input_tensor)
       