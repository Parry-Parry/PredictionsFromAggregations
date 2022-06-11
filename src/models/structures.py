from collections import namedtuple

Dataset = namedtuple(
    'dataset',
    [
        'x_train',
        'x_test',
        'y_train',
        'y_test'
    ])

Config = namedtuple(
    'configuration',
    [
        'in_dim',
        'out_dim'
        'stochastic',
        'lstm'
    ])

Layer = namedtuple(
    'layer',
    [
        'size',
        'activation'
    ])