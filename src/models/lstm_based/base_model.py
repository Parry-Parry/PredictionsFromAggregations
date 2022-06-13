import tensorflow.keras as tfk 
import tensorflow.keras.layers as tfkl
import tensorflow as tf
import logging

class lstm_based(tfk.Model):

  def __init__(self, dim_in, dim_out, config):
    super().__init__()
    self.generator = config.generator
    self.lstm_in = tfkl.lstm(config.lstm.size, activation='relu', return_sequences=True)
    self.lstm_out = tfkl.lstm(config.lstm.size, activation='relu')
    self.output = tfkl.Dense(dim_out, activation='softmax')

  def call(self, inputs):
    x = self.generator(inputs)
    x = self.lstm_in(x)
    x = self.lstm_out(x)

    return self.output(x)

