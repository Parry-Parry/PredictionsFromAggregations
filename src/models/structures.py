from collections import namedtuple

Dataset = namedtuple(
    'dataset',
    [
        'name',
        'x_train',
        'x_test',
        'y_train',
        'y_test'
    ])

Config = namedtuple(
    'configuration',
    [
        'generator',
        'lstm',
        'output'
    ])

Layer = namedtuple(
    'layer',
    [
        'size',
        'activation'
    ])