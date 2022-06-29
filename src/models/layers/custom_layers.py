import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import tensorflow_probability as tfp
import tensorflow_probability.layers as tfpl
import tensorflow as tf

import numpy as np


"""
TODO: 
    Few Shot Tab GAN
    Few Shot Image GAN
    One Shot Variations
    Gumbal Encoder
"""
class gaussian_generator(tfkl.Layer):
    """
    Generate noise from a given shape

    :param int shape: Shape of noise 
    :param np.Array mean: Mean tensor of centroid
    :param np.Array cov: Covariance of centroid
    """
    def __init__(self, shape : int, mean = None, cov = None) -> None:
        super().__init__()
        if mean:
            self.mean = mean
        else:
            self.mean = np.zeros(shape)
        if cov:
            self.cov = cov
        else:
            self.cov = np.eye(shape)

    def _generate_noise(self, tensor):
        return np.random.multivariate_normal(self.mean, self.cov, size=1, check_valid='warn', tol=1e-8) 

    def call(self, inputs):
        return self._generate_noise(inputs)

def _dense_generator(noise_shape : int, latent_dim : int, out_shape : int, i : int):
    x = tfkl.Dense(latent_dim // 2, activation='relu', input_shape=(noise_shape,), name='generator_in_{}'.format(i))
    x = tfkl.Dense(latent_dim, activation='relu', name='generator_latent_{}'.format(i))(x)
    x = tfkl.Dense(out_shape, activation='relu', name='generator_out_{}'.format(i))(x)

    return x
       
class dense_generator(tfkl.Layer):
    """
    Basic autoencoder structure from gaussian noise

    :param int n_generator: The number of samples to be generated from the centroid
    :param int noise_shape: Noise vector size 
    :param int latent_dim: Latent dimension of generator network
    :param int out_shape: Size of original input / output
    """
    def __init__(self, out_shape : int, n_generator=100, noise_shape=100, latent_dim=512) -> None:
        super(dense_generator, self).__init__()
        self.generators = [_dense_generator(noise_shape, latent_dim, out_shape, i) for i in range(n_generator)]
    def call(self, inputs):
        return tf.concat([x(inputs) for x in self.generators], axis=0)


def _dense_reparam_generator(latent_dim : int, out_shape : int, i : int):
    x = tfpl.DenseLocalReparameterization(latent_dim // 2, activation=tf.nn.leaky_relu, input_shape=(out_shape,), name='generator_in_{}'.format(i))
    x = tfpl.DenseLocalReparameterization(latent_dim, activation=tf.nn.leaky_relu, name='generator_latent_{}'.format(i))(x)
    x = tfpl.DenseLocalReparameterization(out_shape, activation=tf.nn.leaky_relu, name='generator_out_{}'.format(i))(x)

    return x
       
class dense_reparam_generator(tfkl.Layer):
    """
    Autoencoder structure using Local Reparameterization (Kingma et. al. 2015) to model distribution from centroid

    :param int n_generator: The number of samples to be generated from the centroid
    :param int latent_dim: Latent dimension of generator network
    :param int out_shape: Size of original input / output
    """
    def __init__(self, out_shape : int, n_generator=100, latent_dim=512) -> None:
        super(dense_reparam_generator, self).__init__()
        self.generators = [_dense_reparam_generator(latent_dim, out_shape, i) for i in range(n_generator)]
    def call(self, inputs):
        return tf.concat([x(inputs) for x in self.generators], axis=0)


class epsilon_generator(tfkl.Layer):
    """
    Generate samples from centroid within its epsilon neighbourhood

    :param int n_generator: The number of samples to be generated from the centroid
    :param float epsilon: Hyperparameter control the size of the neighbourhood
    """
    def __init__(self, out_shape, n_generator=100, epsilon=0.01) -> None:
        super(epsilon_generator, self).__init__()
        self.epsilon = epsilon
        self.n_generator = n_generator

    def _generate_distr(self, tensor):
        return [tfp.distributions.Uniform(low=x-self.epsilon, high=x+self.epsilon) for x in tensor]

    def _sample(self, distr, n):
        squash = lambda x : np.min(np.max(0, x), 1)
        return [tf.constant([squash(d.sample()) for d in distr]) for i in range(n)]

    def call(self, inputs):
        distr = self._generate_distr(inputs)
        samples = self._sample(distr, self.n_generator)
        return tf.concat(samples, axis=0)

class gumbel_generator(tfkl.layer):
    def __init__(self, out_shape, n_generator=100, epsilon=0.01) -> None:
        super(epsilon_generator, self).__init__()
        self.epsilon = epsilon
        self.n_generator = n_generator