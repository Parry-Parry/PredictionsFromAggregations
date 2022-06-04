from responses import activate
from tensorflow.keras import layers, Model

def dense_model(dim_in, dim_out, n_generator, latent_dim):
    inputs = layers.Input(dim_in, name="Input")

    generator_layer = layers.Dense(latent_dim, activation='relu')
    generator_layers = [generator_layer(inputs) for i in range(n_generator)]
    
    
