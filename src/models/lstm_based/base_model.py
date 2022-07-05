import tensorflow.keras as tfk 
import tensorflow.keras.layers as tfkl
import tensorflow as tf

from src.models.structures import *
from src.models.layers.custom_layers import epsilon_generator

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

class epsilon_model(tfk.Model):
    def __init__(self, config : generator_config, epsilon=0.05, name='') -> None:
        super(self, epsilon_model).__init__(name=name)
        self.generators = [epsilon_generator(config.in_dim, config.scale, config.n_classes, i, config.intermediate, epsilon) for i in range(config.n_gen)]
        self.merger = config.merger
        self.out = tfkl.Dense(config.n_classes, activation='softmax')
    def call(self, input_tensor):
        intermediate = tf.concat([gen(input_tensor) for gen in self.generators], axis=0)
        merged = self.merger(intermediate)
        return self.out(merged)
