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
        
class epsilon_generator(tfkl.Layer):
    """
    Generate samples from centroid within its epsilon neighbourhood

    :param int n_generator: The number of samples to be generated from the centroid
    :param float epsilon: Hyperparameter control the size of the neighbourhood
    """
    def __init__(self,  in_dim, scale, n_classes, n, intermediate=None, epsilon=0.05) -> None:
        super(epsilon_generator, self).__init__()
        self.intermediate = intermediate
        self.out = tfkl.Dense(n_classes, activation='softmax', name='generator_out{}'.format(n))
        self.epsilon = epsilon

    def _distr(self, value):
        return tf.math.minimum(tf.math.maximum(0, np.random.uniform(low=value-self.epsilon, high=value+self.epsilon)), 1)

    def _sample(self, tensor):
        spec = tf.TensorSpec(shape=tensor.shape, dtype=tf.uint8)
        return tf.map_fn(self._distr, tensor, fn_output_signature=spec)

    def call(self, input_tensor):
        x = self._sample(input_tensor)
        if self.intermediate: x = self.intermediate(x)
        return self.out(x)