import tensorflow.keras as tfk 
import tensorflow.keras.layers as tfkl
import tensorflow as tf

from keras_multi_head import MultiHead

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
        super(epsilon_model, self).__init__(name=name)
        self.generators = MultiHead(epsilon_generator(config.in_dim, config.scale, config.n_classes, config.intermediate, epsilon), layer_num=config.n_gen, name='Generators')
        self.merger = config.merger
        self.out = tfkl.Dense(config.n_classes, activation='softmax')
    def call(self, input_tensor):
        intermediate_values = self.generators(input_tensor)
        intermediate = tf.reshape(intermediate_values, shape=(intermediate_values.shape[0], )) # MOVE ORIGINAL SHAPE IN HERE
        merged = self.merger(intermediate)
        return self.out(merged)
