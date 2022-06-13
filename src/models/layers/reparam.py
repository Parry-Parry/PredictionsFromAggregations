import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import tensorflow_probability as tfp
import tensorflow_probability.layers as tfpl
import tensorflow as tf

def _dense_reparam_generator(latent_dim : int, out_shape : int, i : int):
    generator = tfk.Sequential()
    generator.add(tfpl.DenseLocalReparameterization(latent_dim // 2, activation=tf.nn.leaky_relu, input_shape=(out_shape,), name='generator_in_{}'.format(i)))
    generator.add(tfpl.DenseLocalReparameterization(latent_dim, activation=tf.nn.leaky_relu, name='generator_latent_{}'.format(i)))
    generator.add(tfpl.DenseLocalReparameterization(out_shape, activation=tf.nn.leaky_relu, name='generator_out_{}'.format(i)))

    return generator
       
class dense_reparam_generator(tfkl.Layer):
    def __init__(self, n_generator : int, latent_dim : int, out_shape : int) -> None:
        self.generators = [_dense_reparam_generator(latent_dim, out_shape, i) for i in range(n_generator)]
    def call(self, inputs):
        return tf.concat([x(inputs) for x in self.generators], axis=0)