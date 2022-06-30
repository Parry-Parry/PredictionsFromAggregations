import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import tensorflow as tf

class generator_block(tfk.Model):
    def __init__(self, in_dim, scale, n_classes, n, intermediate=None) -> None:
        super(generator_block, self).__init__(name='generator{}'.format(n))
        self.dense = tfkl.Dense(128 * scale, input_shape=in_dim, activation='relu')
        self.intermediate = intermediate
        self.out = tfkl.Dense(n_classes, activation='softmax')
        
    def call(self, input_tensor):
        x = self.dense(input_tensor)
        if self.intermediate: x = self.intermediate(x)
        return self.out(x)
        