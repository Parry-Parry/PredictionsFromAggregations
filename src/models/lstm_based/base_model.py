from tensorflow.keras import Model, layers, Input
from collections import namedtuple
import logging

def gen_lstm_model(dim_in, dim_out, config : namedtuple):
    
    stochastic = config.stochastic
    lstm = config.lstm
    out = config.out

    x = Input(dim_in)
    
    for layer in lstm:
        if layer.type == 'lstm':
            x = layers.lstm(layer.main_param, activation = layer.activation, )(x)
        if layer.type == 'dropout':
            x = layers.Dropout(layer.main_param)(x)
        else:
            logging.ERROR("Layer type {} not recognised for this structure, skipping...".format(layer.type))
    x = layers.Dense(dim_out)(x)
    return 
