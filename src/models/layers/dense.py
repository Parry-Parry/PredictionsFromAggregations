import tensorflow.keras as tfk 
import tensorflow.keras.layers as tfkl
import tensorflow as tf

def _dense_generator(noise_shape : int, latent_dim : int, out_shape : int, i : int):
    x = tfkl.Dense(latent_dim // 2, activation='relu', input_shape=(noise_shape,), name='generator_in_{}'.format(i))
    x = tfkl.Dense(latent_dim, activation='relu', name='generator_latent_{}'.format(i))(x)
    x = tfkl.Dense(out_shape, activation='relu', name='generator_out_{}'.format(i))(x)

    return x
       
class dense_generator(tfkl.Layer):
    def __init__(self, n_generator : int, noise_shape : int, latent_dim : int, out_shape : int) -> None:
        self.generators = [_dense_generator(noise_shape, latent_dim, out_shape, i) for i in range(n_generator)]
    def call(self, inputs):
        return tf.concat([x(inputs) for x in self.generators], axis=0)

        

