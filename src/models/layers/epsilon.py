import tensorflow.keras as tfk 
import tensorflow.keras.layers as tfkl
import tensorflow_probability as tfp
import tensorflow as tf
       
class epsilon_generator(tfkl.Layer):
    def __init__(self, n_generator : int, epsilon : float) -> None:
        self.epsilon = epsilon
        self.n_generator = n_generator

    def _generate_distr(self, x):
        distr = []
        for tensor in x:
            lb = tensor - self.epsilon
            ub = tensor + self.epsilon
            distr.append(tfp.distributions.Uniform(low=lb, high=ub))
        return distr

    def _sample(self, distr, n):
        return [tf.constant([d.sample() for d in distr]) for i in range(n)]

    def call(self, inputs):
        distr = self._generate_distr(inputs)
        samples = self._sample(distr, self.n_generator)
        
        return tf.concat(samples, axis=0)
