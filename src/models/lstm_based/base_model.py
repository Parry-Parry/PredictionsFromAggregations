from tensorflow.keras import Model, layers, Input
from collections import namedtuple
import logging

def gen_lstm_model(dim_in, dim_out, config : namedtuple, stochastic=None):
    
    conv = config.conv
    lstm = config.lstm
    out = config.out

    x = Input(dim_in)
    if stochastic:
        x = stochastic(x)
    for layer in conv:
        if layer.type == 'conv2d':
            x = layers.Conv2D(layer.main_param, kernel_size = layer.kernel, activation = layer.activation, paddding = layer.padding)(x)
        if layer.type == 'dropout':
            x = layers.Dropout(layer.main_param)(x)
        else:
            logging.ERROR("Layer type {} not recognised for this structure, skipping...".format(layer.type))
    for layer in lstm:
        if layer.type == 'conv2d':
            x = layers.Conv2D(layer.main_param, kernel_size = layer.kernel, activation = layer.activation, paddding = layer.padding)(x)
        if layer.type == 'dropout':
            x = layers.Dropout(layer.main_param)(x)
        else:
            logging.ERROR("Layer type {} not recognised for this structure, skipping...".format(layer.type))
    x = layers.Dense(dim_out)(x)
    return 
