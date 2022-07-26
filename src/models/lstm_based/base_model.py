import tensorflow.keras as tfk 
import tensorflow.keras.layers as tfkl
import tensorflow.math as tfm
import tensorflow as tf

from src.models.structures import *
from src.models.layers.custom_layers import single_epsilon_generator as epsilon_generator

class lstm_based(tfk.Model):

  def __init__(self, config):
    super(self, lstm_based).__init__()
    self.generator = config.generator
    self.lstm_in = tfkl.Bidirectional(tfkl.lstm(config.lstm.size, activation='relu', return_sequences=True))
    self.lstm_out = tfkl.Bidirectional(tfkl.lstm(config.lstm.size, activation='relu'))
    self.output = tfkl.Dense(config.n_classes, activation='softmax')

  def call(self, inputs):
    x = self.generator(inputs)
    x = self.lstm_in(x)
    x = self.lstm_out(x)
    
    return self.output(x)

class n_epsilon_model(tfk.Model):
    def __init__(self, config : generator_config, epsilon=0.05, name='') -> None:
        super(n_epsilon_model, self).__init__(name=name)
        self.generators = [epsilon_generator(in_dim=config.in_dim, n_classes=config.n_classes, intermediate=config.intermediate, epsilon=epsilon, name=str(n)) for n in range(config.n_gen)]

    def _max_proba(self, proba):
        totals = tfm.reduce_sum(proba, axis=0)
        probs = tf.map_fn(lambda x : tf.one_hot(tf.argmax(x), depth=x.shape[-1], on_value=1, off_value=0), elems=totals)

        return probs

    def call(self, input_tensor, training=False):
        intermediate_values = tf.stack([gen(input_tensor, training) for gen in self.generators], axis=0)
        if training: return self._max_proba(intermediate_values), intermediate_values

        return self._max_proba(intermediate_values)
        

        