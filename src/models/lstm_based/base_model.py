from tensorflow.keras import Model, layers, Input
import tensorflow as tf
import logging

class lstm_based(tf.keras.Model):

  def __init__(self, dim_in, dim_out, config):
    super().__init__()
    self.stochastic = config.stochastic
    self.lstm_in = layers.lstm(config.lstm.size, activation='relu', return_sequences=True)
    self.lstm_out = layers.lstm(config.lstm.size, activation='relu')
    self.output = layers.Dense(dim_out, activation='softmax')

  def call(self, inputs):
    x = self.stochastic(inputs)
    x = self.lstm_in(x)
    x = self.lstm_out(x)

    return self.output(x)

