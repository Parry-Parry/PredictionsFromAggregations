import tensorflow.keras as tfk 
import tensorflow.keras.layers as tfkl
import tensorflow as tf

from keras_multi_head import MultiHead

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

def convnet(in_dim : tuple):
    return tfk.Sequential(
    [
        tfk.Input(shape=in_dim),
        tfkl.Conv2D(32, kernel_size=(3, 3), activation="relu",padding = 'same'),
        tfkl.Conv2D(32, kernel_size=(3, 3), activation="relu",padding = 'same'),
        tfkl.MaxPooling2D(pool_size=(2, 2)),
        tfkl.Dropout(0.25),
        tfkl.Conv2D(64, kernel_size=(3, 3), activation="relu",padding = 'same'),
        tfkl.Conv2D(64, kernel_size=(3, 3), activation="relu",padding = 'same'),
        tfkl.MaxPooling2D(pool_size=(2, 2)),
        tfkl.Dropout(0.25),
        tfkl.Flatten(),
        tfkl.Dense(512, activation='relu')
    ]
)

class epsilon_model(tfk.Model):
    def __init__(self, config : generator_config, epsilon=0.05, name='') -> None:
        super(epsilon_model, self).__init__(name=name)
        self.generators = MultiHead(epsilon_generator(in_dim=config.in_dim, n_classes=config.n_classes, intermediate=config.intermediate, epsilon=epsilon), layer_num=config.n_gen, name='Generators')
        self.out = tfkl.Dense(config.n_classes, activation='softmax')
    def call(self, input_tensor):
        intermediate_values = self.generators(input_tensor)
        merged = tf.math.reduce_mean(intermediate_values, axis=0)
        return self.out(merged)

class n_epsilon_model(tfk.Model):
    def __init__(self, config : generator_config, epsilon=0.05, name='') -> None:
        super(n_epsilon_model, self).__init__(name=name)
        self.generators = [epsilon_generator(in_dim=config.in_dim, n_classes=config.n_classes, intermediate=config.intermediate, epsilon=epsilon, name=str(n)) for n in range(config.n_gen)]
        self.out = tfkl.Dense(config.n_classes, activation='softmax')
    def call(self, input_tensor):
        intermediate_values = tf.stack([gen(input_tensor) for gen in self.generators], axis=0)
        merged = tf.math.reduce_mean(intermediate_values, axis=0)
        return self.out(merged)


class n_epsilon_model(tfk.Model):
    def __init__(self, config : generator_config, epsilon=0.05, name='') -> None:
        super(n_epsilon_model, self).__init__(name=name)
        self.generators = [epsilon_generator(in_dim=config.in_dim, n_classes=config.n_classes, intermediate=config.intermediate, epsilon=epsilon, name=str(n)) for n in range(config.n_gen)]
        self.out = tfkl.Dense(config.n_classes, activation='softmax')
    def call(self, input_tensor):
        intermediate_values = tf.stack([gen(input_tensor) for gen in self.generators], axis=0)
        merged = tf.math.reduce_mean(intermediate_values, axis=0)
        return self.out(merged)
        
class epsilon_3_model(tfk.Model):
    def __init__(self, config : generator_config, epsilon=0.05, name='') -> None:
        super(epsilon_3_model, self).__init__(name=name)
        self.generator1 = epsilon_generator(config.in_dim, config.n_classes, config.intermediate, epsilon)
        self.generator2 = epsilon_generator(config.in_dim, config.n_classes, config.intermediate, epsilon)
        self.generator3 = epsilon_generator(config.in_dim, config.n_classes, config.intermediate, epsilon)
        self.out = tfkl.Dense(config.n_classes, activation='softmax')
    def call(self, input_tensor):
        gen1 = self.generator1(input_tensor)
        gen2 = self.generator2(input_tensor)
        gen3 = self.generator3(input_tensor)

        intermediate = tf.math.reduce_mean([gen1, gen2, gen3], axis=0)
        return self.out(intermediate)

class epsilon_5_model(tfk.Model):
    def __init__(self, config : generator_config, epsilon=0.05, name='') -> None:
        super(epsilon_5_model, self).__init__(name=name)
        self.generator1 = epsilon_generator(config.in_dim, config.n_classes, config.intermediate, epsilon)
        self.generator2 = epsilon_generator(config.in_dim, config.n_classes, config.intermediate, epsilon)
        self.generator3 = epsilon_generator(config.in_dim, config.n_classes, config.intermediate, epsilon)
        self.generator4 = epsilon_generator(config.in_dim, config.n_classes, config.intermediate, epsilon)
        self.generator5 = epsilon_generator(config.in_dim, config.n_classes, config.intermediate, epsilon)
        self.out = tfkl.Dense(config.n_classes, activation='softmax')
    def call(self, input_tensor):
        gen1 = self.generator1(input_tensor)
        gen2 = self.generator2(input_tensor)
        gen3 = self.generator3(input_tensor)
        gen4 = self.generator4(input_tensor)
        gen5 = self.generator5(input_tensor)

        intermediate = tf.math.reduce_mean([gen1, gen2, gen3, gen4, gen5], axis=0)
        return self.out(intermediate)
