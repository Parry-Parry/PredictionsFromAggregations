from responses import activate
from tensorflow.keras import layers, Model

def dense_generator(inputs, n_generator, latent_dim):
    generator_layers = [layers.Dense(latent_dim, activation='relu', name='generator_{}'.format(i))(inputs) for i in range(n_generator)]
    
    
